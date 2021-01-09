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

deletemigrationfiles() {
  find . -path "*/migrations/*.py" -not -name "__init__.py" -delete
  find . -path "*/migrations/*.pyc"  -delete

}

clearmigrations() {
  cd apps/company
  deletemigrationfiles

  cd ../jobs
  deletemigrationfiles

  cd ../pladat
  deletemigrationfiles

  cd ../student
  deletemigrationfiles

  cd ../..
  rm db.sqlite3

  python3 manage.py makemigrations
  python3 manage.py migrate
}