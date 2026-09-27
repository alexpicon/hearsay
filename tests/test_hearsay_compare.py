# Author: Alex Picon <alexnpc@me.com>
"""Security regression tests for the public comparison control plane."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from hearsay.benchmark.dashboard import MODELS, command_for, parse_score
from server.routers import hearsay_compare as api


class ComparisonSecurityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / 'jobs').mkdir()
        key = self.root / 'key'
        key.write_text('test-secret')
        self.patches = [patch.object(api, 'STATE', self.root), patch.object(api, 'KEY', key)]
        for item in self.patches:
            item.start()
        app = FastAPI()
        app.include_router(api.router)
        self.app = app
        self.client = TestClient(app, client=('127.0.0.1', 12345))
        self.headers = {'X-Comparison-Key': 'test-secret', 'X-Models': 'ours', 'X-Audio-Extension': '.wav'}

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.temporary.cleanup()

    def test_unreviewed_image_is_rejected(self):
        from hearsay.benchmark.dashboard import pinned_image
        with patch('hearsay.benchmark.dashboard.docker', return_value='sha256:unreviewed'):
            with self.assertRaises(ValueError):
                pinned_image('ours')

    def test_remote_client_requires_key_and_can_run_with_it(self):
        client = TestClient(self.app, client=('198.51.100.2', 12345))
        self.assertEqual(client.get('/api/hearsay-compare/jobs').status_code, 401)
        self.assertEqual(client.post('/api/hearsay-compare/jobs', content=b'a').status_code, 401)
        self.assertEqual(client.get('/api/hearsay-compare/jobs', headers=self.headers).status_code, 200)
        with patch.object(api.subprocess, 'Popen') as spawn:
            spawn.return_value.pid = 1234
            response = client.post('/api/hearsay-compare/jobs', headers=self.headers, content=b'a')
            self.assertEqual(response.status_code, 202)
            self.assertEqual(spawn.call_count, 1)

    def test_missing_wrong_key_and_forged_forwarded_header_fail(self):
        for headers in [{}, {'X-Comparison-Key':'wrong'}, {'X-Forwarded-For':'127.0.0.1'}]:
            self.assertEqual(self.client.get('/api/hearsay-compare/jobs', headers=headers).status_code, 401)

    def test_cross_origin_fails(self):
        response = self.client.post('/api/hearsay-compare/jobs', headers={**self.headers, 'Origin':'https://evil.example'}, content=b'a')
        self.assertEqual(response.status_code, 403)

    def test_arbitrary_models_and_extensions_rejected(self):
        for change in [{'X-Models':'ours;id'}, {'X-Models':'ours,ours'}, {'X-Audio-Extension':'/../../.env'}]:
            response = self.client.post('/api/hearsay-compare/jobs', headers={**self.headers, **change}, content=b'a')
            self.assertEqual(response.status_code, 400)

    def test_empty_and_oversized_uploads_leave_no_jobs(self):
        for content, status in [(b'', 400), (b'x' * (10 * 1024 * 1024 + 1), 413)]:
            response = self.client.post('/api/hearsay-compare/jobs', headers=self.headers, content=content)
            self.assertEqual(response.status_code, status)
        self.assertEqual(list((self.root / 'jobs').iterdir()), [])

    def test_queue_bound_and_clean_worker_environment(self):
        with patch.object(api.subprocess, 'Popen') as spawn:
            spawn.return_value.pid = 1234
            for _ in range(3):
                response = self.client.post('/api/hearsay-compare/jobs', headers=self.headers, content=b'a')
                self.assertEqual(response.status_code, 202)
            self.assertEqual(self.client.post('/api/hearsay-compare/jobs', headers=self.headers, content=b'a').status_code, 429)
            options = spawn.call_args.kwargs
            self.assertTrue(options['start_new_session'])
            self.assertEqual(set(options['env']), {'PATH', 'HOME', 'DOCKER_HOST', 'DOCKER_CONFIG'})
            self.assertEqual(spawn.call_count, 3)

    def test_display_name_and_optional_label_are_metadata_only(self):
        with patch.object(api.subprocess, 'Popen') as spawn:
            spawn.return_value.pid = 1234
            response = self.client.post('/api/hearsay-compare/jobs', headers={**self.headers,
                'X-Audio-Name':'..%2Fexample%0A.wav', 'X-Expected-Label':'real'}, content=b'a')
            self.assertEqual(response.status_code, 202)
            value = response.json()
            self.assertEqual(value['filename'], 'example.wav')
            self.assertEqual(value['expected_label'], 'real')
            job = self.root / 'jobs' / value['id']
            self.assertEqual([p.name for p in (job/'input').iterdir()], ['clip.wav'])
        bad = self.client.post('/api/hearsay-compare/jobs', headers={**self.headers,
            'X-Expected-Label':'fake;exec'}, content=b'a')
        self.assertEqual(bad.status_code, 400)

    def test_resident_has_no_audio_or_secret_host_mounts(self):
        from hearsay.benchmark.resident import command
        args = command('sha256:example', 'worker-hash')
        mounts = [args[i+1] for i,v in enumerate(args) if v == '--mount']
        self.assertEqual(len(mounts), 1)
        self.assertTrue(mounts[0].endswith('dst=/resident_worker.py,readonly'))
        for flag,value in [('--network','none'),('--memory','6g'),('--cpus','2'),
                           ('--cap-drop','ALL'),('--security-opt','no-new-privileges')]:
            self.assertEqual(args[args.index(flag)+1],value)
        self.assertIn('--read-only',args)
        self.assertNotIn('/var/run/docker.sock', ' '.join(args))

    def test_candidate_mounts_are_readonly_and_contain_no_job_metadata(self):
        from hearsay.benchmark.resident import command
        args = command('sha256:example','hash','ours_v2')
        mounts = [args[i+1] for i,v in enumerate(args) if v=='--mount']
        self.assertEqual(len(mounts),5)
        self.assertTrue(all(m.endswith(',readonly') for m in mounts))
        self.assertFalse(any('result.json' in m or '/jobs/' in m for m in mounts))
        self.assertEqual(args[args.index('--network')+1],'none')
        self.assertEqual(args[args.index('--memory')+1],'10g')
        with self.assertRaises(ValueError):
            command('sha256:example','hash','arbitrary')

    def test_output_requires_exact_id_and_finite_score(self):
        for raw in ['filename\tcm-score\nclip.wav\tnan\n', 'filename\tcm-score\nother.wav\t1\n', 'filename\tcm-score\nclip.wav\t1\nclip.wav\t0\n']:
            with self.assertRaises(ValueError):
                parse_score(raw, 'clip.wav')
        self.assertEqual(parse_score('filename\tcm-score\nclip.wav\t3.2\n', 'clip.wav'), 3.2)

    def test_real_commands_have_only_readonly_host_mounts(self):
        for model in MODELS:
            command = command_for(model, self.root, 'hearsay-security-test')
            for index, part in enumerate(command):
                if part == '--mount':
                    self.assertTrue(command[index + 1].endswith(',readonly'))
                    self.assertNotIn('docker.sock', command[index + 1])
            for flag, value in [('--network','none'), ('--cap-drop','ALL'), ('--security-opt','no-new-privileges'), ('--pull','never')]:
                self.assertEqual(command[command.index(flag) + 1], value)
            self.assertIn('--read-only', command)
            self.assertNotIn('--privileged', command)
            self.assertNotIn('-p', command)


if __name__ == '__main__':
    unittest.main()
