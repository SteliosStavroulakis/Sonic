import retro
import numpy as np
import cv2 
import neat
import pickle


env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')

imgarray = []

def eval_genomes(genomes, config):   #iterate through the steps in the environment

    for genome_id, genome in genomes:
        ob = env.reset()   #observation variable, image, input in NN
        ac = env.action_space.sample()   #just a reneric sample
            
        inx, iny, inc = env.observation_space.shape   #size of image created by emulator

        inx = int(inx/8)
        iny = int(iny/8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config) #this network is what we will eventually use to generate our inputs into the emulator
        
        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        xpos = 0
        xpos_max = 0  #max x position
        
        done = False
        
        while not done:   #while Sonic is alive
            
            env.render()
            frame += 1
            
            ob = cv2.resize(ob, (inx, iny))   #we want fewer input variables, the smaller you can make the # of input variables, the quicker it can find solutions, it just has to be enough information that it can find a solution you want. There is a real balancing act with the implementaion of these NN
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)   #convert to grayscale
            ob = np.reshape(ob, (inx,iny))   #reshape so it fits the neural network
            
            imgarray = np.ndarray.flatten(ob)

            
            nnOutput = net.activate(imgarray)   #create neural net based on input
            
            #print(len(imgarray), nnOutput)
            
            
            
            
            ob, rew, done, info = env.step(nnOutput)   #get new ob, rew, done info. We are literally putting the output of the neural network into this.
            #print(nnOutput)
            
            
            
            
            xpos = info['x']   #grab x position from info variable which contains all of the variables we are recording from the memory addresses from the game.
            
            if xpos > xpos_max:
                fitness_current += 1   #get reward if Sonic goes further to the right
                xpos_max = xpos
                
            #A function that resets a counter each time he hits a new best fitness. This is the same as above, but if you have multiple different variables who are tracking with different functions on how you are recording fitnesses, this would be helpful
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1
            
            if done or counter == 250:   #gets 250 attempts to move right-er
                done = True
                print(genome_id, fitness_current)
                
            genome.fitness = fitness_current   #until we are done, or counter = 250 we are 
            
            

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

p = neat.Population(config) #you setup a config and a population, that generates a bunch of populations, a variable names genomes is create

winner = p.run(eval_genomes)  #calls a function that records the fitness


