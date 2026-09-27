# Author: Alex Picon <alexnpc@me.com>
"""Trusted, bounded Unix-socket scorer inside the offline submission container."""
import json
import os
from pathlib import Path
import signal
import socket
import struct
import sys
import tempfile
import time

SOCKET = '/tmp/hearsay.sock'
MAX_BYTES = 10 * 1024 * 1024
EXTENSIONS = {'.wav', '.flac', '.mp3', '.ogg', '.m4a'}


def receive(connection, count):
    data = bytearray()
    while len(data) < count:
        part = connection.recv(min(count - len(data), 65536))
        if not part:
            raise ValueError('Incomplete request')
        data.extend(part)
    return bytes(data)


def deadline(signum, frame):
    raise TimeoutError('Audio analysis exceeded its deadline')


def serve():
    import torch
    torch.set_num_threads(2)
    if Path('/candidate.json').exists():
        from v2_runtime import build
        score_file = build()
    else:
        from hearsay.inference import get_detector, score_file
        from hearsay.ssl import get_embedder
        detector = get_detector()
        get_embedder(detector.ssl_model, detector.n_layers)
    signal.signal(signal.SIGALRM, deadline)
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(SOCKET)
        os.chmod(SOCKET, 0o600)
        listener.listen(1)
        listener.settimeout(90)
        for _ in range(32):
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                return
            with connection:
                connection.settimeout(15)
                temporary = None
                try:
                    header_size = struct.unpack('!I', receive(connection, 4))[0]
                    if not 0 < header_size <= 256:
                        raise ValueError('Invalid request header')
                    header = json.loads(receive(connection, header_size))
                    size, extension = header['size'], header['extension']
                    if not isinstance(size, int) or not 0 < size <= MAX_BYTES or extension not in EXTENSIONS:
                        raise ValueError('Invalid audio input')
                    with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as handle:
                        temporary = Path(handle.name)
                        handle.write(receive(connection, size))
                    started = time.monotonic()
                    signal.alarm(120)
                    result = score_file(temporary, explain=False)
                    signal.alarm(0)
                    if result.get('error'):
                        raise ValueError('Audio could not be decoded; no score substituted')
                    reply = {'score':result['probability'], 'inference_seconds':time.monotonic()-started,
                             'duration_seconds':result['duration_s'],
                             'analyzed_seconds':result['ssl_analyzed_duration_s'],
                             'coverage_policy':result.get('coverage_policy','first 12 seconds')}
                except Exception as error:
                    reply = {'error':str(error)[:180]}
                finally:
                    signal.alarm(0)
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
                try:
                    connection.sendall(json.dumps(reply, allow_nan=False).encode())
                except (BrokenPipeError, TimeoutError):
                    pass
        # Let the last docker-exec client flush its reply before container exit.
        time.sleep(1)


def client(extension):
    audio = sys.stdin.buffer.read(MAX_BYTES + 1)
    if extension not in EXTENSIONS or not 0 < len(audio) <= MAX_BYTES:
        raise ValueError('Invalid audio upload')
    header = json.dumps({'size':len(audio), 'extension':extension}).encode()
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(135)
        connection.connect(SOCKET)
        connection.sendall(struct.pack('!I',len(header)) + header + audio)
        reply = bytearray()
        while part := connection.recv(4096):
            reply.extend(part)
            if len(reply) > 16384:
                raise ValueError('Oversized model reply')
    sys.stdout.buffer.write(reply)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--client':
        client(sys.argv[2])
    else:
        serve()
