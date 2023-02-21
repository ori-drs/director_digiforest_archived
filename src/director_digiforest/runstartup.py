import director
import sys
import os
import rospkg

sys.path.append(os.path.join(rospkg.RosPack().get_path("director_digiforest"), "src"))
import director_digiforest.startup


director_digiforest.startup.startup(globals())
