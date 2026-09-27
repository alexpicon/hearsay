# Deploy HEARSAY

Author: Alex Picon <alexnpc@me.com>

Production: https://hearsay-ai.tech on `96.30.206.240`. The Atlanta Vultr server
has 6 CPUs, 16 GiB RAM and 320 GiB disk; provisioned pricing is $0.11/hour,
up to $80/month. Charges continue until deletion.

Install Docker, Python, uv, Nginx and Certbot. Put the source at `/opt/hearsay`
and run `uv sync --frozen --no-dev`. Install `hearsay.service` and `nginx.conf`.
The service account is `hearsay`, with Docker group access. Docker access is
privileged; only reviewed controller code should run on the host.

Provide the reviewed images pinned by digest in `hearsay/benchmark/dashboard.py`.
The model cache belongs in `_internal/runtime/assets/hf`; preserve the standard
Hugging Face hub layout and the revisions in `MODEL_CREDITS.md` on the website.
The access code lives in `_internal/runtime/dashboard/access-key` (mode 0600).
Copy `.env` privately with mode 0600; never mount it in model containers.

Only HTTP and HTTPS are public. Restrict SSH to the deployment host in the cloud
firewall. Allow ports 80/443 in UFW. The web service binds port 8888 on loopback.
Point the apex A record to the server and `www` CNAME to the apex. Use Certbot
for both names with HTTP-to-HTTPS redirect and automatic renewal.

Verify `/healthz`, unauthenticated API rejection, authenticated browser uploads
in both modes, and container isolation. The certificate renewal dry run has
passed. The service uses systemd restart-on-failure behavior.
