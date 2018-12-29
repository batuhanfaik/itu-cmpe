#####################################################################
### Assignment skeleton
### You can alter the below code to make your own dynamic website.
### The landing page for assignment 3 should be at /
#####################################################################

from bottle import route, run, default_app, debug, static_file, post, request, get, redirect
from hashlib import sha256

commentList = []
nickList = []

def htmlifyIndex():
    html = """<!DOCTYPE html>
            <html lang="en">
              <head>
                <meta charset="utf-8">

                <title>Derinbay Sailing</title>
                <meta name="description" content="Learn how to sail and physics of sailing.">
                <meta name="author" content="Batuhan Faik Derinbay">

                <link rel="stylesheet" href="css/index_style.css">

                <link rel="shortcut icon" type="image/x-icon" href="img/sailboat_logo_32x32.ico" />
              </head>

              <body>
                <!-- Navbar -->
                <header>
                  <div class="navbar">
                    <a href="">
                      <img src="img/sailboat_logo_v3.svg" class="logo" alt="Logo">
                    </a>

                    <nav>
                      <ul>
                        <li>
                          <a href="">Home</a>
                        </li>
                        <li>
                          <a href="basics_of_sailing/basics.html">The Basics</a>
                        </li>
                        <li>
                          <a href="basics_of_sailing/terms_knots.html">Terms and Knots</a>
                        </li>
                        <li>
                          <a href="basics_of_sailing/firstaid.html">First Aid</a>
                        </li>
                        <li>
                          <a href="basics_of_sailing/regulations.html">Regulations</a>
                        </li>
                      </ul>
                    </nav>
                  </div>
                </header>

                <h1>Sailing for Beginners</h1>

                <img class="image"
                    src="img/blue_gennaker_sailboat.jpg"
                    alt="Sailboath with Blue and White Gennaker"
                    style="width:413px;height:525px;">

                <p>Indeed there are many ways to learn how to sail. You could just hop in a
                  sailboat and sail away or take an online education course. Instead
                  you are here reading our article.</p>
                <p>Well then. Welcome abord Cabin Boy!</p>
                <p>From now on we will teach you the basics of sailing.
                  <br>You should <em style="color:red">follow these steps in order</em>, otherwise harsh
                  seas might get you!
                  <br>Let's start with the basics.</p>

                <h3>Basics of Sailing</h3>
                <ol>
                  <li><b>Sailing Terms</b>
                      <br>Terms used in sailing are very important. In order to communicate effectively sailors often use these terms. <a href="basics_of_sailing/terms_knots.html#terms">Here</a>
                      you can review some of the terms we've selected for you.
                  </li>
                  <li><b>Parts of the Boat</b>
                      <br>You need to learn some equipment to use those terms right? <a href="basics_of_sailing/basics.html#parts">Here</a> you can see the basic outline of a sail and
                      its parts.
                  </li>
                  <li><b>Physics of Sailing</b>
                      <br>Now that you know the terms and your sailing equipment, let's see how sailboats actually travel on bodies of water. You really don't need a high understanding of
                      physics. All you need to know is <em>Vectors</em>. Follow <a href="http://newt.phys.unsw.edu.au/~jw/sailing.html" target="_blank">this</a> page for more detailed information on physics of
                      sailing.
                  </li>
                  <li><b>Knots</b>
                      <br>The best friend of a sailor is his knowledge of knots. The more you know the better. They help the skipper in the most needed times by stopping sails from flying
                      away or patching a hole. Such essential knots and their areas of use can be found <a href="basics_of_sailing/terms_knots.html#knots">here</a>. By the way, patching holes with the
                      help of knots is not exaggeration. You only need to be creative from time to time.
                  </li>
                  <li><b>Getting the Boat and Yourself Ready</b>
                      <br>What equipment does the boat need? Well, its sails and ropes, you say. But, is the boat only thing that requires equipment? Do you think you are ready for the next
                      sailing season? Let's see how to set the boat up and get you ready for the season. <a href="basics_of_sailing/basics.html#equipment">Here</a>, you will find what you
                      and your loved one (your boat) need.
                    </li>
                  <li><b>Sailing Techniques</b>
                      <br>In <a href="basics_of_sailing/basics.html#techniques">this</a> section, you will be learning basically how to sail. How to trim your sails according to the wind,
                      which angle you should attack the wind from etc. Your understanding of this section is pretty important for advancement in sailing.
                  </li>
                  <li><b>Maneuvering and Getting to a Point</b>
                      <br>This section is pretty intimate with sailing techniques. Uses of your sail trimming and rudder handling determine how fast you maneuver and your turn radius. It is
                      also important to note that not all sailing is done for performance purposes. Sometimes you just want to get to a specific location. So depending on where you want to
                      go with your sailboat, you can read <a href="http://www.schoolofsailing.net/navigation.html" target="_blank">this</a> page on how get there as efficiently as possible.
                  </li>
                  <li><b>Docking and Anchoring</b>
                      <br>You got to your destination yet? That's fantastic. But, wait a second. You can't just leave your boat like that. You have to dock or anchor it. Don't worry we got
                      you covered. <a href="basics_of_sailing/basics.html#docking">Here</a> you can learn more on how to leave your sailboat safely and properly.
                  </li>
                  <li><b>Safety Onboard</b>
                      <br>Sea, sun, swimming, sunbathing, enjoying the view etc. All seem exiciting right? Well things might not always go as planned. Unexpected weather conditions or
                      injuries might occur during the next time you got onboard. You should always remember <a href="basics_of_sailing/firstaid.html#firstaid">these</a> quick safety tips
                      carry a first aid kit at all times.
                  </li>
                  <li><b>Rules, Regulations and Maritime Codes</b>
                      <br>Just like traffic on roads, there is traffic on seas. Therefore, vessels have some regulatory rules and codes for navigation and safety. While these are many, and
                      most of them are aimed towards other boats like cargo ships or fisherman, <a href="basics_of_sailing/regulations.html#regulations">this</a> section will include the ones
                      that basic sailors should know and obey. Don't forget that without regulations, your safety will be in trouble. Therefore always follow the rules accordingly.
                  </li>
                </ol>
                <div class="is_this">
                  <a href="https://jigsaw.w3.org/css-validator/validator?uri=https%3A%2F%2Fituis18.github.io%2Fa1-batuhanfaik%2Findex.html&profile=css3svg&usermedium=all&warning=1&vextwarning=&lang=en" target="_blank"><img src="img/valid_css.jpg" alt="Valid CSS" title="Is This a Valid CSS?"/></a>
                  <a href="https://validator.w3.org/nu/?doc=https%3A%2F%2Fituis18.github.io%2Fa1-batuhanfaik%2Findex.html" target="_blank"><img src="img/valid_html.jpg" alt="Valid HTML" title="Is This a Valid HTML?"/></a>
                </div>
                <form action="/" method="post">
                    Nickname: <input name="nick" type="text" />
                    Anonymous: <input name="anon" type="checkbox" />
                    <br /><br />
                    Comment: <input name="comment" type="text" />
                    Password: <input name="password" type="password" />
                    <input value="Submit" type="submit" />
                </form>
                <h2>Comments</h2>
                {}
              </body>
            </html>
            """.format(htmlify_ulist(nickList, commentList))
    return html

def htmlify_ulist(nickList, commentList):
    listString = "<ul>"
    nickCount = 0
    for listItem in commentList:
        listString += "<li>{}: {}</li>".format(nickList[nickCount], listItem)
        nickCount += 1
    listString += "</ul>"
    return listString

#This function was copied from "https://bitbucket.org/damienjadeduff/hashing_example/raw/master/hash_password.py"
def create_hash(password):
    pw_bytestring = password.encode()
    return sha256(pw_bytestring).hexdigest()

#My password stored as hash
password_hash = "7ca6e264880d73246ffc076f15b42a2aa5857021e4f3beb06c3c83332ce59722"

# @get('/')
# def login():
#     return htmlifyIndex()

@route('/')
def index():
    return htmlifyIndex()

@post('/')
def do_login():
    global commentList
    global nickList
    nick = request.forms.get('nick')
    anon = request.forms.get('anon')
    comment = request.forms.get('comment')
    password = request.forms.get('password')
    if anon == "on":
        nick = "Anonymous"
    nickList.insert(0, nick)
    if create_hash(password) == password_hash:
        commentList.insert(0, comment)
    redirect('/')


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

