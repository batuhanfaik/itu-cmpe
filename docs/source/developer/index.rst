Developer Guide
===============

Database Design
---------------

Our database is in 1NF since some of our team members decided to use `TR ID` as a primary
key for their relations while others preferred an serial integer.

Also there is a table named `syllabus` which has the same primary key `course` does. This `file` attribute
in `syllabus` table is separated for practical reasons. Since multiple querries occur on the course table that
does not use syllabus pdf, downloading the pdf file would have decrease system speed.

E/R Diagrams of DataBees
++++++++++++++++++++++++

.. figure:: ../../images/databees_erd_derinbay.png
    :alt: ERD of Derinbay
    :align: center

.. figure:: ../../images/faati_erd.png
    :alt: ERD of Altınpınar
    :align: center

.. figure:: ../../images/zeynepyetistiren_ERD.png
    :alt: ERD of Yetistiren
    :align: center

.. figure:: ../../images/akkiraz/cihat_erd.png
    :alt: ERD of Akkiraz
    :align: center
    
Code
----

ITU DataBees integrates the **model-view-controller** design.

- **Web Framework:** Flask
- **Database:** PostgreSQL
- **Written in:** Python

.. figure:: ../../images/mvc.png
    :alt: Model-View-Controller
    :align: center

    Model-View-Controller

ITU DataBees is structed as follows:

- ``server.py`` creates a flask app, handles the incoming requests and routes them.
- ``views.py`` renders the templates.
- ``database.py`` handles the requests between the database and the web application.
- ``settings.py`` has the settings for the flask web application.
- ``dbinit.py`` initializes the DataBees.
- ``forms.py`` handles the forms controlled by flask forms.
- Various class named files includes corresponding python classes and related functions.
- Templates folder contains the HTML files of the websites
- Static folders contains the images, style sheets and scripts used during the development of the DataBees.

.. toctree::

   Batuhan Faik Derinbay
   fatih
   Zeynep Yetistiren
   Cihat Akkiraz
