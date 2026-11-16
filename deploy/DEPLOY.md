# Putting NeuroScope online at your own domain

The result: `https://yourdomain` opens NeuroScope, with HTTPS (lock icon) set up automatically.
Total time: about 45 minutes, most of it waiting.

## What you need

| Item | Where | Cost |
|---|---|---|
| GitHub Student Developer Pack | education.github.com/pack (verify with your SRM email) | Free |
| A domain, e.g. `neuroscope.tech` | Student Pack offer, or buy `.in` / `.com` on Namecheap, GoDaddy, Hostinger | Free to about Rs 1,000 a year |
| A server | DigitalOcean (Student Pack credit) | Covered by credit for months |

Check the current Student Pack offers and DigitalOcean prices before you sign up; they change.

## 1. Create the server (DigitalOcean)

1. Sign in to DigitalOcean and apply the Student Pack credit.
2. **Create > Droplets**.
3. Region: **Bangalore (BLR1)**.
4. Image: **Ubuntu 24.04 LTS**.
5. Size: Basic, Regular, **4 GB RAM / 2 CPUs**. 2 GB is too small for MRI volumes.
6. Authentication: **Password** is the simplest. Pick a strong one and keep it.
7. Create. Copy the **IP address** it shows, e.g. `139.59.12.34`.

## 2. Point your domain at the server

In your domain provider's **DNS settings**, add two records:

| Type | Name / Host | Value |
|---|---|---|
| A | `@` | your server IP |
| A | `www` | your server IP |

Changes usually take 5 to 30 minutes to work.

## 3. Copy the model and example patient to the server

On your laptop, in PowerShell (replace the IP):

```
cd "C:\Users\Neeti Soni\neuroscope"
scp -r runs root@139.59.12.34:/root/
scp -r demo_case root@139.59.12.34:/root/
```

Type `yes` if asked, then the server password. `runs` must contain `runs\base\best.pt` and `runs\base\results`.

## 4. Set up the server

Still in PowerShell:

```
ssh root@139.59.12.34
```

Then on the server, paste these one block at a time:

```
curl -fsSL https://get.docker.com | sh
ufw allow OpenSSH && ufw allow 80 && ufw allow 443 && ufw --force enable
git clone https://github.com/neetii-git/neuroscope.git
mv /root/runs /root/demo_case /root/neuroscope/
cd /root/neuroscope/deploy
```

Start the site (replace with your domain, without `www` or `https`):

```
DOMAIN=neuroscope.tech docker compose up -d --build
```

The first build takes about 5 to 10 minutes. Then open `https://neuroscope.tech`.

## Updating the site after changes

```
ssh root@139.59.12.34
cd /root/neuroscope && git pull
cd deploy && DOMAIN=neuroscope.tech docker compose up -d --build
```

## If something does not work

| Problem | Check |
|---|---|
| Site does not open | `docker compose ps` should show `app` and `caddy` as running. `docker compose logs app` shows errors. |
| Browser warns about the certificate | DNS has not updated yet. Wait 15 minutes, then `docker compose restart caddy`. |
| "Model file not found" | `ls /root/neuroscope/runs/base/best.pt` must exist. |
| Example patient missing | `ls /root/neuroscope/demo_case` must list the `.nii` files. |

## Privacy

Uploaded scans are read through a temporary file that is deleted immediately, then held in memory for the visitor's session only. Nothing is kept after the session ends.
The site is a research and education tool, not a medical device; the disclaimer stays visible in the app and the report.
