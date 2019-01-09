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
You can use a local installation or a hosted service like
`ElephantSQL <https://www.elephantsql.com/>`_ (they have a free plan),
but we recommend that you use `Docker <https://www.docker.com/>`_::

  $ docker pull postgres

In order to make changes to your database persistent, you have to set up
a folder that will be shared between your regular operating system and
the Docker container. Create the folder, e.g.::

  $ mkdir -p $HOME/docker/volumes/postgres

The command for running the container is::

  $ docker run --rm --name pg-docker -e POSTGRES_PASSWORD=docker -d -p 5432:5432 -v $HOME/docker/volumes/postgres:/var/lib/postgresql/data postgres

This will start a PostgreSQL server that runs on the host ``localhost``,
on port 5432. The username is ``postgres``, the password is ``docker``,
and the database name is ``postgres``. You can use the following command
to connect to it::

  $ psql -h localhost -U postgres -d postgres

You should arrange the ``dbinit.py`` script to properly initialize
your database. This script requires that you provide the database URL
as an environment variable, so here's an example of how you can run it::

  $ DATABASE_URL="postgres://postgres:docker@localhost:5432/postgres" python dbinit.py

**Documentation**

The documentation template is located under the ``docs/source`` folder.
Change the project name "ITUCSDB18NN" in the ``conf.py`` file to match
your team name.

`Travis-CI <https://travis-ci.org/>`_ will be used to automatically
publish your Sphinx documentation on Github Pages:

- Visit the https://github.com/settings/tokens page and click on
  "Generate new token". Select the ``public_repo`` permission,
  generate the token, and copy it to the clipboard.

- Enable your project on Travis. (*ITU students*: only one team member
  needs to do this).

- In the Travis-CI project settings, create a new environment variable
  in the "Environment Variables" section. Set the variable name as
  ``GH_TOKEN`` and paste your token as the value. The option
  "Display value in build log" should be ``off``.

- After pushing your changes to the ``master`` branch, your documentation
  will become updated on ``https://itucsdb18NN.github.io/itucsdb18NN/``
