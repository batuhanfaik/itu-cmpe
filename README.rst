This repository contains a template project for the Database Systems course
of the Istanbul Technical University Computer Engineering Department.
The project uses the Python language, the Flask web application framework,
and the PostgreSQL database.

How to use this repository
--------------------------

**ITU students**

Create a GitHub organization for your team using the name that was assigned
to your team (in the form ``itucsdb18NN`` where ``NN`` is your team number)
and fork this repository into that organization. Rename your repository
so that it will have the same name as the team. Then every team member
has to clone the repository to their local machines::

  $ git clone git@github.com:itucsdb18NN/itucsdbNN.git

**Setup**

Run the following command to install the dependencies::

  $ pip install -r requirements.txt

You can now start the application using the command::

  $ python server.py

And when you visit the address http://localhost:5000/ you should see
the "Hello, world!" message.

Alternatively, you can also start the application using the command::

  $ gunicorn server:app

In this case, the address will be http://localhost:8000/

**Database**

By default, the project is meant to be used with a PostgreSQL server.
You can use any PostgreSQL installation but a Dockerfile is provided
for convenience. To build the container, run::

  $ docker build -t itucsdb .

The command for running the container is::

  $ docker run -P --name postgres itucsdb

If you have a PostgreSQL client you can connect to the server using
the username ``itucs``, the password ``itucspw`` and the database
``itucsdb``. The server will be accessible through the host ``localhost``
but you have to figure out the port number::

  $ docker ps
  CONTAINER ID        IMAGE               COMMAND                  CREATED             STATUS              PORTS                     NAMES    
  d58b35f6503b        itucsdb             "/usr/lib/postgresql…"   8 minutes ago       Up 8 minutes        0.0.0.0:32775->5432/tcp   postgres

In this example, under the ``PORTS`` column, you can see that the port number
is ``32775``.

If you don't have a PostgreSQL client, you can use another docker instance::

  $ docker run -it --rm --link postgres:postgres postgres psql -h postgres -U itucs itucsdb

You should arrange the ``dbinit.py`` script to properly initialize
your database. This script requires that you provide the database URL
as an environment variable, so here's an example of how you can run it::

  $ DATABASE_URL="postgres://itucs:itucspw@localhost:32775/itucsdb" python dbinit.py

.. image:: https://www.herokucdn.com/deploy/button.svg
   :alt: Deploy to Heroku
   :target: https://heroku.com/deploy
