#!/bin/bash

export APP_NAME=$1

rm /etc/nginx/sites-enabled/default
cp /app/nginx/default /etc/nginx/sites-enabled/default

nginx &
cd /app/mhcfovea
gunicorn mhcfovea.wsgi --bind=unix:/tmp/gunicorn.sock --worker-class=sync --workers=5 --timeout 900

sleep infinity
