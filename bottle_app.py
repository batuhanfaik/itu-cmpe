
#####################################################################
### Assignment skeleton
### You can alter the below code to make your own dynamic website.
### The landing page for assignment 3 should be at /
#####################################################################

from bottle import route, run, default_app, debug, static_file
import indexHTML

# def htmlify(title,text):
#     page = """
#         <!doctype html>
#         <html lang="en">
#             <head>
#                 <meta charset="utf-8" />
#                 <title>%s</title>
#             </head>
#             <body>
#             %s
#             </body>
#         </html>
#
#     """ % (title,text)
#     return page
#
# def index():
#     return htmlify("My lovely website",
#                    "This is going to be an awesome website, when it is finished.")

@route('/') #same thing with --> route('/', 'GET', index)
def index():
    return indexHTML.htmlifyIndex()

#HTML Routing
@route('/basics_of_sailing/<htmlPath>')
def html(htmlPath):
    return static_file(htmlPath, root="./basics_of_sailing")

#CSS Routing
@route('/css/<cssPath>')
def css(cssPath):
    return static_file(cssPath, root="./css")

#Media and Asset Routing
@route('/img/<imgPath>')
def img(imgPath):
    return static_file(imgPath, root="./img")
@route('/img/backgrounds/<bg>')
def imgBg(bg):
    return static_file(bg, root="./img/backgrounds")
@route('/img/knots/<knot>')
def imgKnot(knot):
    return static_file(knot, root="./img/knots")
@route('/img/rules/<rule>')
def imgRule(rule):
    return static_file(rule, root="./img/rules")
@route('/img/stars/<star>')
def imgStar(star):
    return static_file(star, root="./img/stars")

#Icon Routing (Doesn't work doe)
@route('/img/sailboat_logo_32x32.ico')
def logoIcon():
    return static_file('sailboat_logo_32x32.ico', root='./img')


#####################################################################
### Don't alter the below code.
### It allows this website to be hosted on Heroku
### OR run on your computer.
#####################################################################

# This line makes bottle give nicer error messages
debug(True)
# This line is necessary for running on Heroku
app = default_app()
# The below code is necessary for running this bottle app standalone on your computer.
if __name__ == "__main__":
  run(reloader=True)

