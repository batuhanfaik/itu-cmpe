This repository contains a template project for the Database Systems course
of the Istanbul Technical University Computer Engineering Department.
The project uses the Python language, the Flask web application framework,
and the PostgreSQL database.

How to use this repository
--------------------------

*ITU students*: Create a GitHub organization for your team and
fork this repository into that organization. Rename your repository
to the name assigned to your team (as in ``itucsdb1899``).

Run the following command to install the dependencies::

  pip install -r requirements.txt

You can now start the application using the command::

  python server.py

And when you visit the address http://localhost:5000/ you should see
the "Hello, world!" message.

Alternatively, you can also start the application using the command::

  gunicorn server:app

In this case, the address will be http://localhost:8000/

.. image:: https://www.herokucdn.com/deploy/button.svg
   :alt: Deploy to Heroku
   :target: https://heroku.com/deploy
