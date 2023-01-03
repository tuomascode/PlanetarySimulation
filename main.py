# Project description:

# Three body problem animation done with Python pygame.

# Features:
#     1. three body objects that are spheres
#         i) Size, mass and speed vectors
#         ii) physics simulation via Newtons laws(ish) 
#     2. Animation without infinite parallel lines
#         i) Animation is build on three mathemathical objects. The center of the celestial system, a focal_vector which determines the where the center is viewed
#            and a drawing plane
#         ii) The animation class allos the manipulation of plane distance and focal_vector, which in essense allow the movement of the camera
#         iii) Sphere objects allow for animation of sphere by drawing a lot of polygons. Currently they are drawn as a bunch of dots #TODO
#     3. Other features
#         i) pause button
#         ii) reset system
#         iii) Randomize system


import random,pygame
import numpy as np
from collections import deque

class Sphere:
    #This class holds the functionality to draw the sphere
    def __init__(self,radius,angle_count=7):
        self.color=np.random.randint(0, 256, size=3)
        while np.sum(self.color)<500:
            index=random.randint(0,2)
            self.color[index]+=5
            if self.color[index]>255:
                self.color[index]=255
        #Phase is the index which we use to get the wanted vertices out of phase_list.
        #This helps us rotate the sphere without calculating the rotations separetely.
        self.phase=0
        self.phase_list=[]
        self.turn_angle=0

        #Define the angle of rotations
        self.angle_count=angle_count

        #The following code calculates the vertices. Later I will use the rotate function to get the rest

        vertices=[]    #Holds the vertices. Later we will for the polygons out of these
        vector=np.array([radius,0,0])   #Initial vector for calculating
        self.radius=radius
        #Get the middle section of the sphere vertices
        vertices.append(self.generate_vertices(vector,self.angle_count))

        #Calculate the distance we want to between two points in the sphere
        self.fixed_distance=self.distance(vertices[0][1],vertices[0][0])

        #Rotating the vector in z axis in order to get the new level of vertices
        rotated_vector=self.rotate_z_axis(vector,self.angle_count)

        #This list will hold the negative values of the vertices, so I don't have to solve them twice.
        #Basically this just mirrors the top of the sphere
        negative_y_list=[]

        #Rotating the vector in z-axis will decrease its x value, untill it flips to below zero.
        #No need to continue after that
        while rotated_vector[0]>=0:

            #Employing the fancy binary search to get the new angle so the distance between the points remains the same
            new_angle=self.solve_angle(self.fixed_distance,rotated_vector)

            #Using the same function as before to get the new vertices
            new_list=self.generate_vertices(rotated_vector,new_angle)

            #Get the flipped coordinates as well.
            negative_y_list.append(self.flip_list(new_list))
            vertices.append(new_list)
            
            #Rotate the vector at the end
            rotated_vector=self.rotate_z_axis(rotated_vector,self.angle_count)

        #And now adding the flipped list and original list together. Also correcting the order. Now each first order list is a list of coordinates at lowest level up to the 
        #highest level
        negative_y_list.reverse()
        vertices=negative_y_list+vertices

        self.phase_list.append(vertices)

        #Here I use rotate_sphere function to save the rest of the phases of the rotation of the sphere.

        for i in range(self.angle_count-1):
            self.rotate_sphere(1)
    def rotate_sphere(self,angle):
        #Basically a copy of the init function but with a twist. The initial vector is rotated to create a spinning effect.
        self.turn_angle+=angle
        if self.turn_angle>self.angle_count-1:
            self.turn_angle=0

        vertices=[]

        vector=np.array([self.radius,0,0])
        if self.turn_angle!=0:
            vector=self.rotate_y_axis(vector,self.turn_angle)
        vertices.append(self.generate_vertices(vector,self.angle_count))

        self.fixed_distance=self.distance(vertices[0][1],vertices[0][0])

        rotated_vector=self.rotate_z_axis(np.array([self.radius,0,0]),self.angle_count)
        old_angle=self.angle_count

      
        negative_y_list=[]

      
        while rotated_vector[0]>=0:

            new_angle=self.solve_angle(self.fixed_distance,rotated_vector)

            rotated_vector=self.rotate_y_axis(rotated_vector,new_angle/old_angle*self.turn_angle)
            old_angle=new_angle

            new_list=self.generate_vertices(rotated_vector,new_angle)

            negative_y_list.append(self.flip_list(new_list))
            vertices.append(new_list)
            

            rotated_vector=self.rotate_y_axis(rotated_vector,-new_angle/old_angle*self.turn_angle)

            rotated_vector=self.rotate_z_axis(rotated_vector,self.angle_count)

        negative_y_list.reverse()
        vertices=negative_y_list+vertices
        self.phase_list.append(vertices)
    def transform_list(self,coord_list):
        #This is for testing that the vertices are calculated correctly

        # Create an empty list to store the transformed coordinates
        single_list = []
        
        # Iterate through the lists of coordinates
        for coords in coord_list:
            # Create an empty list to store the single coordinate list
            
            # Iterate through the coordinates
            for coord in coords:
                # Add the coordinate to the single list
                single_list.append(coord)
        return single_list
    def adjust_color(self,z_value, radius):
        # Calculate the new value
        new_value =self.color * ((z_value*-1) / radius)**2
        #To get colorful sphere set disco value to True
        disco=False
        if disco:
            value_one,value_two,value_three = new_value,new_value,new_value

            if random.randint(0,1)==1:
                value_one=new_value-random.randint(-100,100)
                value_one=min(max(value_one, 0), 255)
            if random.randint(0,1)==1:
                value_two=new_value-random.randint(-100,100)
                value_two=min(max(value_one, 0), 255)
            if random.randint(0,1)==1:
                value_three=new_value-random.randint(-100,100)
                value_three=min(max(value_one, 0), 255)
            return value_one,value_two,value_three
        else:
            # Return a tuple of three values equal to the new value
            # print(new_value)
            return new_value
    def adjust_radius(self,z_value, radius,size_factor):
        # Calculate the new value
        new_value = -1*(self.radius//15) * z_value / radius
        
        # Round the value to the nearest integer, convert and return
        return int(round(new_value*size_factor))
    def flip_list(self,list):
        #This copies the original list and changes the positivity of the y value, or flipping the vector

        #I love me some numpy operations <3
        flipped_list=np.copy(list)
        #By multiplying all the y values by -1, I solve half of the sphere vertices
        flipped_list[:,1] *= -1
        return flipped_list
    def draw_dots(self, screen,size_factor=1,position=(400,400),rotation=True):
        #Just iterating through the vertices to draw them as spots. This is just for testing that the vertice coordinates are
        #calculated correctly


        for vertex in self.transform_list(self.phase_list[self.phase]): #Get the right phase or rotation from memory
            # Extract the x and y coordinates from the vertex
            x = vertex[0]
            y = vertex[1]
            
            # Convert the coordinates to integers
            x = int(x)*size_factor+position[0]
            y = int(y)*size_factor+position[1]
            # Draw the dot using Pygame's draw.circle() function

            #Adjust the color to reflect depth

            if False:
                pygame.draw.circle(screen, self.adjust_color(vertex[2],self.radius), (x, y), self.adjust_radius(vertex[2],self.radius,size_factor))
            else:
                pygame.draw.circle(screen, self.adjust_color(vertex[2],self.radius), (x, y), 1)
        if rotation:
            self.phase+=1
            if self.phase>self.angle_count-1:
                self.phase=0
    def generate_vertices(self,vector,angle):
        #Test function for now to try and get the middle level vertices of a sphere

        #Initial vector that points to the direction of x axis
        vertices=[vector]

        #Rotating the vector while it's z component is positive. This should get me all the vertices I need.
        vector=self.rotate_y_axis(vector,angle)
        while vector[2]<=0:
            vertices.append(vector)
            vector=self.rotate_y_axis(vector,angle)
        return vertices
    def rotate_z_axis(self,vector, angle):
        # Convert the angle to radians
        theta = np.radians(angle)
        
        # Define the rotation matrix
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        
        # Rotate the vector
        rotated_vector = np.dot(rotation_matrix, vector)
        
        return rotated_vector
    def rotate_y_axis(self,vector, angle):
        # Convert the angle to radians
        theta = np.radians(angle)
        
        # Define the rotation matrix
        rotation_matrix = np.array([[np.cos(theta), 0, np.sin(theta)],
                                    [0, 1, 0],
                                    [-np.sin(theta), 0, np.cos(theta)]])
        
        # Rotate the vector
        rotated_vector = np.dot(rotation_matrix, vector)
        
        return rotated_vector
    def distance(self,coords1, coords2):
        # Calculate the difference between the coordinates
        diff = coords1 - coords2
        
        # Calculate the distance using the Euclidean distance formula
        dist = np.sqrt(np.sum(diff ** 2))
        
        return dist
    def solve_angle(self,fixed_distance,vector):

        # Set the initial values for the binary search
        low = 0
        high = 180

        # Set the tolerance for the distance error
        tolerance = 1e-6

        # Perform the binary search
        while high - low > tolerance:
            # Calculate the midpoint angle
            mid = (low + high) / 2
            
            # Rotate the vector around the y-axis
            rotated_vector =self. rotate_y_axis(vector, mid)
            
            # Calculate the distance between the original and rotated vectors
            dist = self.distance(vector, rotated_vector)
            
            # Check if the distance is too small or too large
            if dist < fixed_distance:
                low = mid
            else:
                high = mid

        # Print the final angle
        return mid
    def create_polygons(self):
        #This functions takes the coordinates of the vertices and creates polygons out of them
        #Here the idea is to use to indices technque to get all the polygons.
        #

        pass
class Body:
    #This is class for the celestial bodies. This class will hold basic parameters and feature functions
    def __init__(self,random_size=False):
        # A celestial body has a mass, size, position and speed and direction 
        self.size=random.randint(85,95)
        if random_size:
            self.size=random.randint(60,120)
        #Using the imported sphere for later animation
        self.sphere=Sphere(self.size)
        #Density will determine the mass of the sphere
        self.density=50_000
        self.color=(random.randint(100,255),random.randint(100,255),random.randint(100,255))
        self.mass=self.size**3*np.pi*(4/3)*self.density
        #Just some random coordinates. Doesn't matter since these are never used
        self.coordinates=[random.randint(-200,1400) for i in range(3)]
        if self.coordinates[2]<400:
            self.coordinates[2]=400+random.randint(0,200)
        self.coordinates=np.array(self.coordinates).astype(np.float64)
        self.speed_vector=[0 for i in range(3)]
        self.speed_vector=np.array(self.speed_vector).astype(np.float64)
        
        #Defining a list to track history of celestial body. This is to draw a trail behind it
        self.history=deque()
    def update_history(self):
        #History is used to draw the dots behind the planets. Makes the animation 9000 powerpoints cooler. Change the value in while loop to get some more tail baby.
        self.history.append(self.coordinates.copy())
        while len(self.history)>200:
            self.history.popleft()
    def sun(self,center):
        #Fun "sunnifier" function. Make one planet very large and see that happens. Altho this function isn't used anywhere at the moment.
        self.size=150
        self.mass=self.size**3*np.pi*(4/3)*0.1
        self.coordinates=center
        self.color=(255,255,255)
        self.speed_vector=[0 for i in range(3)]
        self.speed_vector=np.array(self.speed_vector).astype(np.float64)
    def __str__(self) -> str:
        #Just to get basic info of the body before pygame visualization functions are complete
        string_to_return+=f"My coordinates are: {self.coordinates}\n"
        string_to_return+=f"My speed vector is: {self.speed_vector}"
        return string_to_return
    def move(body_a, body_b):
        #Calculate the gravitational force and apply the effects
        acc_a,acc_b=Body.calculate_gravitational_acceleration(body_a.coordinates,body_b.coordinates,body_a.mass,body_b.mass)
        body_a.speed_vector+=acc_a
        body_b.speed_vector+=acc_b
        body_a.coordinates+=body_a.speed_vector
        body_b.coordinates+=body_b.speed_vector
    def calculate_gravitational_acceleration(a_coords, b_coords, a_mass, b_mass):
        #Give this function two celestial bodies and you shall receive some acceleration values. Hurraah!

        # Calculate the distance between the objects
        distance = np.linalg.norm(b_coords - a_coords)
        
        # Calculate the gravitational constant
        G = 6.67430 * 10**-11
        
        # Calculate the gravitational acceleration of each object
        a_acceleration = G * b_mass / distance**2 * (b_coords - a_coords)
        b_acceleration = G * a_mass / distance**2 * (a_coords - b_coords)
        
        return a_acceleration, b_acceleration
class Animation:
    #This class holds a variety of functions required to animate the three body system.
    #In essence, the Animation class hold these functions:
    #   1. Calculating the position of geometric center of the three body system and focal point
    #   2. Manipulating the focal point and viewing plane with focal_vector, or turn the camera angle in common tongue
    #   3. Calculating intersection points of viewing plane and body-focal point lines
    #   4. Normalizing the intersection coordinates to the viewing plane coordinates
    #   5. Drawing functions for x-y-z normal coordinate lines and the bodies
    #   6. Rotating functions for moving focal_point and vectors

    def __init__(self,bodies):
        #This class will hold the paramaters and functions that are required to animate the three body system

        #Calculate center, define focal_point and inital plane_factor, which is used to determine the position of the drawing plane
        self.focal_vector=np.array([0,0,-800])

        #special_body is used for the blackhole feature. This allows the program to use one of the bodies as the center of the system.
        self.special_body=bodies[1]
        self.update_center(bodies)
        self.focal_point=self.solve_focal_point()
        self.plane_factor=0.5
    def update_body_locations(self,three_body_set):
        #This functions updates the locations of the bodies using newtonian math-physics-stuff.
        three_body_set[0].move(three_body_set[1])
        three_body_set[0].move(three_body_set[2])
        three_body_set[1].move(three_body_set[2])
    def update_center(self,bodies):
        #Function that updates the center location


        if not Game.blackhole:
            #Here the program offers a choice. Use weighted averaged of the system or just geometric average.
            if Game.weighted_average:
                #Weighted average
                self.center=self.get_weighted_average([i.coordinates for i in bodies],[i.mass for i in  bodies])
            else:
                #This is the usual way to do solve the geometric center. 
                self.center=bodies[0].coordinates.copy()
                self.center+=bodies[1].coordinates
                self.center+=bodies[2].coordinates
                self.center/=3
        else:
            self.center=self.special_body.coordinates.copy()

        self.focal_point=self.solve_focal_point()
    def zoom(self,value):
        #The plane factor determines the location of the viewing plane. Zoom function just changes that value

        if value!=1:
            self.plane_factor*=1.1
        else:
            self.plane_factor/=1.1
    def solve_focal_point(self):
        #Eases the conceptual work to have this function.

        #First we copy the center. Copying to make sure changes in the center don't have effects on this vector
        vector=self.center.copy()
        #The focal_vector is a directional vector which tells where to go from center to reach the focal_point. This is the math for that
        #Thank god for numpy and array wise operations
        vector+=self.focal_vector
        return vector
    def resize_vector(self,vector, length):
        # Fun vector resizer, who wouldn't want a piece of code like this? Could be used to change the focal_vector to manipulate camera
        # Calculate the current length of the vector
        current_length = np.linalg.norm(vector)
        
        # Calculate the scaling factor
        scale = length / current_length
        
        # Scale the vector
        scaled_vector = vector * scale
        
        return scaled_vector
    def write_coords_to_screen(self,screen):
        #This functions can be used to track positions of center, plane point and the distance between center and focal point
        #Center refers to geometric center of the three body system
        #This can be used for other reporting stuff on the pygame screen
        #I used this for testing to get a sense of the math behind

        font = pygame.font.SysFont("Times New Roman", 24)
        writing="Center: "+str(self.center)
        text = font.render(writing, True, (255, 255, 255))
        screen.blit(text, (25, 955))
        writing="Plane point: "+str(self.get_plane_point())
        text = font.render(writing, True, (255, 255, 255))
        screen.blit(text, (25, 925))
        writing="Focal point distance: "+str(self.get_distance_of_points(self.center,self.focal_point))
        text = font.render(writing, True, (255, 255, 255))
        screen.blit(text, (25, 895))
        writing="Focal point: "+str(self.focal_point)
        text = font.render(writing, True, (255, 255, 255))
        screen.blit(text, (25, 865))
    def get_distance_of_points(self,vector_one,vector_two):
        #Called 8 times in the program, I think. Noice.

        # Calculate the difference between the coordinates
        diff = vector_one-vector_two
        
        # Calculate the distance using the Euclidean distance formula
        dist = np.sqrt(np.sum(diff ** 2))
        
        return dist
    def get_weighted_average(self,positions,masses):
        #Give this function three coordinates and masses and it will give you the average weighted geometric position

        #Get total mass
        mass_sum = np.sum(masses)
        #Define the sum variable
        sum_coord=np.array([0,0,0]).astype(np.float64)
        #Be lame and do this with for loop like a noob.
        for coord,mass in zip(positions,masses):
            sum_coord+=(coord*mass)
        #Divide the result with total mass
        sum_coord/=mass_sum
        #Send back the results
        return sum_coord
    def get_perpendicular_vector(self,v1,v2):
        #This function returns a vector that is at 90 deg angle to both v1 and v2
        #This corresponds to the y-vector of the drawing plane.

        a=v1[1]*v2[2]-v1[2]*v2[1]
        b=v1[2]*v2[0]-v1[0]*v2[2]
        c=v1[0]*v2[1]-v1[1]*v2[2]
        return [a,b,c]
    def get_normal_vectors(self,vector):
        #These functions gets vectors that are perpendicular, match the drawing plane and are 'straight'
        #in the sense that the drawing plane doesn't get rotated.

        x_vector = np.array([vector[2],0,-vector[0]]).astype(np.float64)
        y_vector = np.array(self.get_perpendicular_vector(x_vector,vector)).astype(np.float64)

        #Get unit vector by dividing by distance
        x_vector/=self.get_distance_of_points(x_vector,[0,0,0])
        y_vector/=self.get_distance_of_points(y_vector,[0,0,0])
        return np.array(x_vector).astype(np.float64),np.array(y_vector).astype(np.float64)
    def get_plane_point(self):
        #This function gets the plane point and conviniently makes sure that the zoom outward doesnt pass the focal point
        #Not sure if this is good coding practice but meh, it'll do for this project

        #
        vector_from_center=-self.focal_point+self.center
        vector_from_center*=self.plane_factor
        while self.get_distance_of_points(vector_from_center,[0,0,0])>800:
            self.zoom(1)
            vector_from_center=self.focal_point-self.center
            vector_from_center*=self.plane_factor
        return self.focal_point+vector_from_center
    def get_body_plane_intersections(self,bodies):

        #This piece of code can be a little hard to read. The idea is to get normalized coordinates for the drawing plane to draw the celestial bodies


        #First lets define the lines between the bodies and the focal point
        line_one=Line(-self.focal_point+bodies[0].coordinates,self.focal_point)
        line_two=Line(-self.focal_point+bodies[1].coordinates,self.focal_point)
        line_three=Line(-self.focal_point+bodies[2].coordinates,self.focal_point)
        #Define the drawing plane
        get_plane_point=self.get_plane_point()
        plane_normal_vector=-get_plane_point+self.center
        drawing_plane=Plane(plane_normal_vector,get_plane_point)

        #Solve the x,y,z values of the intersections. Three values for three celestial bodys
        cord_one,cord_two,cord_three= drawing_plane.intersect(line_one),drawing_plane.intersect(line_two),drawing_plane.intersect(line_three)


        #Normalize the coordinates by removing the get_plane_point values out of vectors. Now we can calculate their value from 0,0,0 position. Neat
        get_plane_point=get_plane_point.reshape(3,1)
        cord_one-=get_plane_point
        cord_two-=get_plane_point
        cord_three-=get_plane_point

        #Solve the x and y vectors that allow us to solve the drawing coordinates
        x_vector,y_vector=self.get_normal_vectors(plane_normal_vector)

        #And solve the differential equations. Basically asking how many x and y vectors do I need to get to the point of intersection.
        m_list=np.array([x_vector,y_vector,[0,0,0]]).reshape(3,3)
        body_one_intersection=m_list.dot(cord_one.reshape(3,1))[0:2]
        body_two_intersection=m_list.dot(cord_two.reshape(3,1))[0:2]
        body_three_intersection=m_list.dot(cord_three.reshape(3,1))[0:2]

        #Return just the the x and y values. z values are not needed
        return [body_one_intersection[0:2],body_two_intersection[0:2],body_three_intersection[0:2]]
    def get_relative_size(self,bodies):
        #Presuming that relative size for drawing is a linear function. Getting the relative sizes I should draw.
        #To be perfectly honest this bit of code is a little shady. Should make sure this is correct math #TODO

        sizes=[]
        for body in bodies:
            size=body.size*(self.get_distance_of_points(self.get_plane_point(),self.focal_point))
            size/=self.get_distance_of_points(body.coordinates,self.focal_point)
            sizes.append(size)
        return sizes
    def get_absolute_xyz_line_points(self):
        #If you run the program you see x y and z directional lines. These lines give points of reference to the system.
        #The lines are drawn by calculating where their relative ends are in the drawing plane. Similar code solves the drawing coordinates for
        #celestial bodies and their histories (the lines behind them) 


        # Define points and drawing plane
        point_one,point_two,point_three=np.array([700,0,0]),np.array([0,700,0]),np.array([0,0,700])
        get_plane_point=self.get_plane_point()
        plane_normal_vector=-get_plane_point+self.center
        drawing_plane=Plane(plane_normal_vector,get_plane_point)

        #Calculating the absolute values of the intersections
        intersections=[]

        #Note the pro for loop from 1 to -1. If i multiply 700,0,0 by -1, I get -700,0,0 which is the other opposite coordinate I need!
        for i in [1,-1]:
            #Define the Line objects that are between focal_point and the end of our reference line.
            line_one=Line(-self.focal_point+i*point_one+self.center,i*point_one+self.center)
            line_two=Line(-self.focal_point+i*point_two+self.center,i*point_two+self.center) 
            line_three=Line(-self.focal_point+i*point_three+self.center,i*point_three+self.center)
            #Solve the intersection points in the system.
            cord_one,cord_two,cord_three= drawing_plane.intersect(line_one),drawing_plane.intersect(line_two),drawing_plane.intersect(line_three)
            get_plane_point=get_plane_point.reshape(3,1)
            #Negating the plane point is essential for the math. Easier to calculate how many x and y vectors do I need to get from
            # 0,0,0 to somewhere else, than from some random values such as 1.5 , 301.99 , -100
            cord_one-=get_plane_point
            cord_two-=get_plane_point
            cord_three-=get_plane_point
            intersections.append([cord_one,cord_two,cord_three])

        #Calculate the normalized coordinates or the drawing plane coordinates
        normalized_coordinates=[]

        #Get the normalized unit vectors
        x_vector,y_vector=self.get_normal_vectors(plane_normal_vector)

        #Solve differential equation. Basically ask, how many normalized x and y vectors do I need to reach the intersection.
        m_list=np.array([x_vector,y_vector,[0,0,0]]).reshape(3,3)
        for a,b,c in intersections:
            norm_cord_one=m_list.dot(a.reshape(3,1))[0:2]
            norm_cord_two=m_list.dot(b.reshape(3,1))[0:2]
            norm_cord_three=m_list.dot(c.reshape(3,1))[0:2]
            normalized_coordinates.append([norm_cord_one,norm_cord_two,norm_cord_three])
        #The return form is weird but necessary to order the points for my function.
        return [[normalized_coordinates[0][0],normalized_coordinates[1][0]],[normalized_coordinates[0][1],normalized_coordinates[1][1]],[normalized_coordinates[0][2],normalized_coordinates[1][2]]]
    def get_normal_coordinates(self,coordinates):
        #Define the list which will hold the no
        normal_coordinates=[]
        #Define the drawing plane
        get_plane_point=self.get_plane_point()
        plane_normal_vector=-get_plane_point+self.center
        drawing_plane=Plane(plane_normal_vector,get_plane_point)
        get_plane_point=get_plane_point.reshape(3,1)
        #Get normal vectors that allow us to solve the normal coordinates
        x_vector,y_vector=self.get_normal_vectors(plane_normal_vector)
        #Use this for solving
        m_list=np.array([x_vector,y_vector,[0,0,0]]).reshape(3,3)
        for coord in coordinates:
            if self.same_side_of_plane(coord,self.focal_point,drawing_plane):
                #Let's not draw points that are not opposite to the focal point
                continue
            #Define the line between focal_point and the historical coordinate of the body
            line=Line(-self.focal_point+coord,self.focal_point)
            #Solve the intersection point in x,y,z
            cord_one=drawing_plane.intersect(line)-get_plane_point
            #Solve the intersection in normal values, lose the extra dimension and add to list
            intersection=m_list.dot(cord_one.reshape(3,1))[0:2]
            intersection=[intersection[0][0],intersection[1][0]]
            normal_coordinates.append(intersection)
        #Return the solved coordinates
        return normal_coordinates
    def get_x_z_angles(self,coords):
        # Convert the coordinates to a numpy array
        coords = np.array(coords)

        
        # Calculate the distance from the origin
        distance = np.linalg.norm(coords)
        
        # Calculate the angles in radians
        x_angle = np.arccos(coords[0] / distance)
        z_angle = np.arccos(coords[1] / distance)
        
        # Convert the angles to degrees
        x_angle = np.degrees(x_angle)
        z_angle = np.degrees(z_angle)
        
        return x_angle, z_angle
    def draw_lines_and_bodies(self,screen,three_body_set,rotation=True):

        #First use the previous functions to get values we need
        normalized_coordinates=self.get_body_plane_intersections(three_body_set)
        sizes=self.get_relative_size(three_body_set)
        line_points=self.get_absolute_xyz_line_points()
        
        #Define the draw_list. This one gets functions and the values it needs to draw on the screen.
        #I use this format because by writing a sort function I can determine the draw order correctly. Not done yet tho #TODO
        draw_list=[]
        #Add coordinates and functions for drawing lines
        for a,b in line_points:
            cx1,cy1=a[0]+600,a[1]+500
            cx2,cy2=b[0]+600,b[1]+500
            draw_list.append((pygame.draw.line,(screen,(55,55,55),(cx1[0],cy1[0]),(cx2[0],cy2[0]))))
        
        
        #Sort the bodies. First is furthest from the focal_point
        three_body_set.sort(key=lambda x: -self.get_distance_of_points(self.focal_point,x.coordinates))
        get_plane_point=self.get_plane_point()
        plane_normal_vector=-get_plane_point+self.center
        drawing_plane=Plane(plane_normal_vector,get_plane_point)

        #Iterate over the necessary info to draw. A rare zip function use for me.
        for coordinates,body,size in zip(normalized_coordinates,three_body_set,sizes):
            #Check on which side of plane the body is:
            if self.same_side_of_plane(body.coordinates,self.focal_point,drawing_plane):
                #Makes sense to skip drawing stuff that is on the same side of the drawing plane as the focal_point
                continue

            #Crab the coordinates and add the half of screen width and height to values.
            x,y=600+coordinates[0][0],500+coordinates[1][0]

            #Draw black circle for some reason.
            #TODO might have to get rid of this when I'm doing the polygon spheres
            #Draw a dot sphere. Kinda cool but will be replaced by polygons.
            draw_list.append((pygame.draw.circle,(screen,(0,0,0),(x,y),size)))
            draw_list.append((body.sphere.draw_dots,(screen,size/body.size,(x,y),rotation)))
        
        #I absolutely love this. By looking at just this code you would never know what is going on.
        #This in fact does almost all the drawing required
        for function, values in draw_list:
            function(*values)
    def draw_histories(self,screen,bodies):
        #Draw the dots behind the planets. Very cool

        #Iterate over bodies
        for body in bodies:
            #Get all the drawing coordinates at one go. Wow
            coords=self.get_normal_coordinates(list(body.history))
            
            #Iterate over the coordinates to draw on the screen. Use the same color as the sphere to make it sexy.
            for x,y in coords:
                x+=600
                y+=500
                pygame.draw.circle(screen,body.sphere.color,(x,y),1)
    def rotate_y_axis(self,angle,vector):
        #Simple function to rotate a vector around the y-axis
        angle=np.deg2rad(angle)
        rotation_matrix=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        new_vector_one=np.matmul(rotation_matrix,vector)
        return new_vector_one
    def rotate_x_axis(self,angle,vector):
        angle=np.deg2rad(angle)
        rotation_matrix=np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
        new_vector_one=np.matmul(rotation_matrix,vector)
        return new_vector_one
    def rotate_z_axis(self,angle, vector):

        angle = np.deg2rad(angle)
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],[np.sin(angle), np.cos(angle), 0],[0, 0, 1]])
        new_vector_one=np.matmul(rotation_matrix,vector)
        return new_vector_one
    def rotate_focal_point_xz(self,angle):
        #Since the up-down rotation is most of the time a composite rotation, a lot of checking has to be done

        #Solve the angle of axises and the focal point
        coords=self.focal_point-self.center
        coords=[coords[0],coords[2]]
        x_angle,z_angle=self.get_x_z_angles(coords)
        #normalize them between 0 - 2
        x_angle/=90
        z_angle/=90

        #These checks ensure the right kind of rotations are done depending on which quadrant the camera is.
        #I figured this out with trial and error basically
        #There is not problem remaining. If you are just to right of one of the axes, the up-down motion doesn't work. My guess is that one value
        #should change it's sign to work... #TODO

        if x_angle<=1:
            if z_angle<=1:
                #Lets just explain this one as an example.

                #Since I want Z-rotation to be 0 when z_angle is 1 and 1 when z_angle is 0, I can flip the values with this formula
                z_angle=1-z_angle
                #Now just rotate the focal_vector by the given values. X-axis rotation needs to be negated, otherwise the rotations cancel each other out.
                #I just tested the camera rotations until I found the right combination of negative one multipliers.
                new_vector=self.rotate_x_axis(-angle*x_angle,self.focal_vector)
                new_vector=self.rotate_z_axis(angle*z_angle,new_vector)
                #Update the focal_vector.
                self.focal_vector=new_vector
            else:
                z_angle-=1
                z_angle=1-z_angle
                new_vector=self.rotate_x_axis(angle*x_angle,self.focal_vector)
                new_vector=self.rotate_z_axis(angle*z_angle,new_vector)
                self.focal_vector=new_vector
        else:
            if z_angle<=1:
                x_angle-=1
                x_angle=1-z_angle
                new_vector=self.rotate_x_axis(-angle*x_angle,self.focal_vector)
                new_vector=self.rotate_z_axis(-angle*z_angle,new_vector)
                self.focal_vector=new_vector
            else:
                x_angle-=1
                x_angle=1-x_angle
                z_angle-=1
                new_vector=self.rotate_x_axis(angle*x_angle,self.focal_vector)
                new_vector=self.rotate_z_axis(-angle*z_angle,new_vector)
                self.focal_vector=new_vector
    def rotate_focal_point_y(self,angle):
        self.focal_vector=self.rotate_y_axis(angle,self.focal_vector)
    def same_side_of_plane(self,coords1, coords2, plane):
        # Calculate the signed distances from the plane for both points
        distance1 = np.dot((coords1.reshape(3,1) - plane.point_on_plane.reshape(3,1)).reshape(3), plane.normal_vector.reshape(3))
        distance2 = np.dot((coords2.reshape(3,1) - plane.point_on_plane.reshape(3,1)).reshape(3), plane.normal_vector.reshape(3))
        # If either of the distances is zero (the point is on the plane), return True
        if distance1 == 0 or distance2 == 0:
            return True
        
        # If the distances have the same sign, the points are on the same side of the plane
        return (distance1 > 0) == (distance2 > 0)

class Plane:
    #Plane and Line classes exist to calculate the intersection between drawing plane and
    #lines between the bodies and focal point
    def __init__(self, normal_vector, point_on_plane) -> None:
        self.normal_vector = np.array(normal_vector).reshape(3, 1)
        self.point_on_plane = np.array(point_on_plane).reshape(3, 1)
    def intersect(self, line: 'Line'):
        if self.normal_vector.ravel().dot(line.vector.ravel()) != 0:
            d = (self.point_on_plane - line.point_on_line).ravel().dot(
                self.normal_vector.ravel()) / self.normal_vector.ravel().dot(
                    line.vector.ravel())
            return line.point_on_line + (d * line.vector)
        return None
class Line:
    def __init__(self, vector, point_on_line) -> None:
  
        self.vector = np.array(vector).reshape(3, 1)
        self.point_on_line = np.array(point_on_line).reshape(3, 1)
    def __str__(self) -> str:
        return f"{self.vector}\n{self.point_on_line}"
class MenuScreen:
    #This class is to holds features and functions to draw on the screen instructions
    def __init__(self) -> None:
        self.show_instructions=True
        #Saving the information for drawing the boxes and texts in dict variables.
        self.continue_box={"font":pygame.font.SysFont("Times New Roman", 36),"box_pos":(25,25),"text_pos":(27,27),"color":(180,180,180),"dimensions":(220,50),"text":"Continue (esc)"}
        self.reset_box={"font":pygame.font.SysFont("Times New Roman", 36),"box_pos":(260,25),"text_pos":(227+35,27),"color":(180,180,180),"dimensions":(150,50),"text":"Reset (r)"}
        self.random_box={"font":pygame.font.SysFont("Times New Roman", 36),"box_pos":(260+160,25),"text_pos":(227+35+160,27),"color":(180,180,180),"dimensions":(220,50),"text":"Randomize (u)"}
    #The draw functions are quite self-explanatory. Just draw instructive stuff on the screen. Some boolean action if the user wants to hide instructions.
    def draw_pause_box(self,game,menu):
        if self.show_instructions:
            if menu:
                pygame.draw.rect(game.screen,self.continue_box["color"],(self.continue_box["box_pos"],self.continue_box["dimensions"]))
                game.screen.blit(self.continue_box["font"].render("Continue (esc)", True, (0,0,0)),self.continue_box["text_pos"])
            else:
                game.screen.blit(self.continue_box["font"].render("Pause (esc)", True, (55,55,55)),self.continue_box["text_pos"])
    def draw_reset_box(self,game):
        if self.show_instructions:
            pygame.draw.rect(game.screen,self.reset_box["color"],(self.reset_box["box_pos"],self.reset_box["dimensions"]))
            game.screen.blit(self.reset_box["font"].render(self.reset_box["text"], True, (0,0,0)),self.reset_box["text_pos"])
    def draw_random_box(self,game):
        if self.show_instructions:
            pygame.draw.rect(game.screen,self.random_box["color"],(self.random_box["box_pos"],self.random_box["dimensions"]))
            game.screen.blit(self.random_box["font"].render(self.random_box["text"], True, (0,0,0)),self.random_box["text_pos"])
    def draw_instructions(self,game):
        if self.show_instructions:
            text="To move the camera angle, click and drag with mouse"
            pygame.draw.rect(game.screen,self.random_box["color"],((25,100),(435,80)))
            game.screen.blit(pygame.font.SysFont("Times New Roman", 18).render(text, True, (0,0,0)),(27,102))
            text="To zoom in and out of the system, use mouse wheel"
            game.screen.blit(pygame.font.SysFont("Times New Roman", 18).render(text, True, (0,0,0)),(27,127))
            text="(b) Blackhole, (a) weighted average center, (h) instructions"
            game.screen.blit(pygame.font.SysFont("Times New Roman", 18).render(text, True, (0,0,0)),(27,127+25))

class Game:
    #The Game class hold basic game functions such as running the simulation and resetting the system with new variables.


    #One class boolean for one feature only. In the blackhole mode I want the camera to center around the 'blackhole'. 
    #This variable allows my other classes to look up to this boolean and determine the center of the system.
    blackhole=False
    weighted_average=True
    def __init__(self):
        #Define the basic pygame stuff
        pygame.init()
        self.tickrate=50
        self.screen_size=1200,1000
        self.screen=pygame.display.set_mode(self.screen_size)
        self.clock=pygame.time.Clock()
        self.screen_color=10,10,10
        #Using the reset function here to save me from coding that stuff twice.
        self.reset()
        self.menu=MenuScreen()
        self.blackhole=False
        pygame.time.set_timer(pygame.USEREVENT,50)
        pygame.display.set_caption("Celestial Body Simulation")

    def reset(self):
        Game.blackhole=False
        self.tickrate+=10
        #Create bodies and give them some initial conditions.
        #If you want to see more interesting movements, you should play with these values.

        #The system center should be around 500,500,500. Move bodies away from that and give them some speed values to see what changes in the behauviour of the system.
        #Note that masses and sizes are still randomized a bit.
        self.three_body_set=[Body() for i in range(3)]
        self.three_body_set[0].coordinates=np.array([500+600,600,500]).astype(np.float64)
        self.three_body_set[1].coordinates=np.array([500,400,500]).astype(np.float64)
        self.three_body_set[2].coordinates=np.array([500-600,500,500]).astype(np.float64)
        self.three_body_set[0].speed_vector[2]+=2.3
        self.three_body_set[2].speed_vector[2]-=2.3
        self.three_body_set[0].speed_vector[1]+=0.3
        self.three_body_set[2].speed_vector[1]-=0.3
        
        #Define the animation class for focal_pointy,drawing_plany, math and rotation stuff. Fun!
        self.animation=Animation(self.three_body_set)
    def randomize(self):
        self.tickrate=50
        Game.blackhole=False
        #Basically like reset but with more goody randomness to make exiting new systems.

        self.three_body_set=[Body(random_size=True) for i in range(3)]
        self.three_body_set[0].coordinates=np.array([500+600,600,500]).astype(np.float64)
        self.three_body_set[1].coordinates=np.array([500,400,500]).astype(np.float64)
        self.three_body_set[2].coordinates=np.array([500-600,500,500]).astype(np.float64)
        for body in self.three_body_set:
            for index in range(len(body.speed_vector)):
                body.speed_vector[index]+=(random.random()-0.5)*6
        self.animation=Animation(self.three_body_set)
    def black_hole(self):
        self.three_body_set=[Body() for i in range(3)]
        self.three_body_set[0].coordinates=np.array([500+600,600,500]).astype(np.float64)
        self.three_body_set[1].coordinates=np.array([500,400,500]).astype(np.float64)
        self.three_body_set[2].coordinates=np.array([500-600,500,500]).astype(np.float64)
        self.three_body_set[0].speed_vector[2]+=20.3
        self.three_body_set[2].speed_vector[2]-=20.3
        self.three_body_set[0].speed_vector[1]+=0.3
        self.three_body_set[2].speed_vector[1]-=0.3
        self.three_body_set[1].size=100
        self.three_body_set[1].density=3_000_000
        self.three_body_set[1].mass=self.three_body_set[1].size**3*np.pi*(4/3)*self.three_body_set[1].density
        self.three_body_set[1].sphere=Sphere(10)
        self.three_body_set[1].size=10
        Game.blackhole=True

        self.three_body_set[1].sphere.color=np.array([255,255,255])

        #Define the animation class for focal_pointy,drawing_plany, math and rotation stuff. Fun!
        self.animation=Animation(self.three_body_set)
    def simulate(self):
        #Basic pygame style code to run the game.

        rotate=False
        menu=False
        while True:
            for event in pygame.event.get():
                if event.type==pygame.USEREVENT:
                    for body in self.three_body_set:
                        if not menu:
                            body.update_history()
                #Mouse wheel for moving drawing plane
                if event.type==pygame.MOUSEWHEEL:
                    if event.y==-1:
                        self.animation.zoom(1)
                    else:
                        self.animation.zoom(-1)
                #Mouse dragging in x and y directions to rotate the focal_point, or move the camera in normal-people-talk
                if event.type==pygame.MOUSEBUTTONDOWN:
                    if event.button==1:
                        rotate=True
                        x1,y1=pygame.mouse.get_pos()
                if event.type==pygame.KEYDOWN:
                    #Access the menu screen with escape
                    if event.key==pygame.K_ESCAPE:
                        if menu:
                            menu=False
                        else:
                            menu=True
                    #r button to reset the simulation
                    if event.key==pygame.K_r:
                        self.reset()
                        menu=False
                    #u button to randomize the simulation
                    if event.key==pygame.K_u:
                        self.randomize()
                        menu=False
                    #Make one of the planets a blackhole with b
                    if event.key==pygame.K_b:
                        self.black_hole()
                        menu=False
                    #Show instructions h - key
                    if event.key==pygame.K_h:
                        if self.menu.show_instructions:
                            self.menu.show_instructions=False
                        else:
                            self.menu.show_instructions=True
                    #Switch between weighted and geometric average using a key
                    if event.key==pygame.K_a:
                        if Game.weighted_average:
                            Game.weighted_average=False
                        else:
                            Game.weighted_average=True
                #stop rotation when click is no longer clickety (let go off mouse button)
                if event.type==pygame.MOUSEBUTTONUP:
                    rotate=False
                #Need a button to quit the fancy simulation
                if event.type == pygame.QUIT:
                    exit()
            #Some code to move the gamera
            if rotate:
                x,y=pygame.mouse.get_pos()
                self.animation.rotate_focal_point_xz(-(y1-y)/5)
                self.animation.rotate_focal_point_y(-(x1-x)/10)
                x1,y1=pygame.mouse.get_pos()

            self.screen.fill(self.screen_color)
            #Calculate center and focal point
            self.animation.update_center(self.three_body_set)
            #Keep moving the planets when menu is not open. Menu also functions as a pause. When menu is open, show buttons and instructions. Hide instructions to look at the pretty planets.
            if not menu:
                self.animation.update_body_locations(self.three_body_set)
            else:
                self.menu.draw_reset_box(self)
                self.menu.draw_random_box(self)
                self.menu.draw_instructions(self)
            self.animation.draw_histories(self.screen,self.three_body_set)

            #One big monster function. Need to simplify this to make it comprehensible #TODO
            self.animation.draw_lines_and_bodies(self.screen,self.three_body_set,rotation=False)
            # self.animation.test_same_side(self.three_body_set)
            #Draw pause box :)
            self.menu.draw_pause_box(self,menu)
            pygame.display.flip()
            self.clock.tick(self.tickrate)
def run_game():
    game=Game()
    game.simulate()
if __name__=="__main__":
    run_game()
