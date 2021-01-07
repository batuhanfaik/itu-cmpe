#!/bin/env bash

runserver() {
  python3 manage.py runserver 0.0.0.0:8000
}

makemigrations() {
  python3 manage.py makemigrations
}

migrate() {
  python3 manage.py migrate
}