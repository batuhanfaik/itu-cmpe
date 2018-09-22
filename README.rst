This repository contains a template project for the Database Systems course
of the Istanbul Technical University Computer Engineering Department.
The project uses the Python language, the Flask web application framework,
and the PostgreSQL database.

How to use this repository
--------------------------

**ITU students**

- Create a GitHub organization for your team using the name that was assigned
  to your team (in the form ``itucsdb18NN`` where ``NN`` is your team number).

- Fork this repository *into that organization*.

- Rename the repository so that it will have the same name as the team.

- (Every team member) Clone the repository to your local machine.

**Setup**

Run the following command to install the dependencies::

  $ python -m pip install -r requirements.txt

You can now start the application using the command::

  $ python server.py

And when you visit the address ``http://localhost:5000/`` you should see
the "Hello, world!" message.

Alternatively, you can also start the application using the command::

  $ gunicorn server:app

In this case, the address will be ``http://localhost:8000/``.

**Deploying to Heroku**

- Use the button below to deploy your application to Heroku.
  (*ITU students*: Only one team member needs to do this.)

  .. image:: https://www.herokucdn.com/deploy/button.svg
     :alt: Deploy to Heroku
     :target: https://heroku.com/deploy

- As "App name" enter your team name, and click the "Deploy app" button.

- Now when you click the "View" button you should access your application
  under the address ``http://itucsdb18NN.herokuapp.com/``.

We want to set up our project so that whenever commits are pushed
to the GitHub repo the application will be automatically deployed to Heroku.

- Click the "Manage app" button and choose the "Deploy" tab.

- For deployment method, choose "GitHub (Connect to GitHub)"
  and under its options choose "Automatic deploys from GitHub".

- If requested, allow Heroku to access your repositories.
  (*ITU students*: In the authorization form, also grant access
  to the organization repositories before submitting the form.)

- Choose your repository, click the "Connect" button, and then
  click the "Enable automatic deploys" button.

In your code, change the "Hello, world!" message, and commit and push
your change to GitHub. A while later the application on Heroku should
display the new message. You can use the activity tab on Heroku to see
how deployments are going.

**Database**

By default, the project is meant to be used with a PostgreSQL server.
You can use any PostgreSQL installation but a Dockerfile is provided
for convenience. Docker will host the PostgreSQL client under the hood.
Docker is a container for your programs which allows 
you to unify development/testing/production environments.

In MacOS, you can install Docker from its official website
https://www.docker.com/products/docker-desktop

Many Linux distributions has Docker in its official package repositories.
https://docs.docker.com/install

Before running Docker, you may need to start Docker service in Linux. This can be done by many ways,
one is explained below:
https://docs.docker.com/install/linux/linux-postinstall
It is quite advanced for new Linux user but it is manageble.

Another options is to start Docker Daemon manually, 
Which can be achieved by one of following commands:

* $ ``sudo systemctl start docker``
* $ ``sudo service docker start``
* $ ``sudo dockerd``

After making sure that Docker daemon is up and running, you are ready to build/start containers.

To build the container, run:: (You may need to have root privelege for using Docker depending on Docker Daemon. If it is the case, add ``sudo`` to commands.)

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

**Deploying Documentation**

Documentation is located in `docs/source` directory.
You should change `ITUCSDB18NN` in `conf.py` file to match your team name.

Travis-CI will be used to automatically publish Sphinx documentation in Github Pages.

- Create a token by visiting https://github.com/settings/tokens page and clicking on "Generate new token".
- Select `public_repo` permission, generate the token and copy it to clipboard.
- Enable your project in https://travis-ci.org (*ITU students*: only one team member needs to do this).
- In project settings (in Travis-CI) add your token in "Environment Variables" section.
- Set variable name as `GH_TOKEN` and paste your token to the value. "Display value in build log" should be `off`.
- After pushing anything to the `master` branch, your documentation will become visible at https://itucsdb18NN.github.io/itucsdb18NN/
