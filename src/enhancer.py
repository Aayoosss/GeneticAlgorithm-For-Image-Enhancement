import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
import os

log_dir_path = "logs"
os.makedirs(log_dir_path, exist_ok = True)


logger = logging.Logger("enhancer.py")
logger.setLevel("DEBUG")

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel("DEBUG")

file_path = os.path.join(log_dir_path,"enhancer.log")
fileHandler = logging.FileHandler(file_path)
fileHandler.setLevel("DEBUG")

Formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fileHandler.setFormatter(Formatter)
consoleHandler.setFormatter(Formatter)

logger.addHandler(consoleHandler)
logger.addHandler(fileHandler)

class Enhancer():
    def __init__(self, image : np.array, cliplimit: float):
        try:
            self.image = image
            self.max_m = image.shape[0]//4
            self.min_m = 2
            self.max_n = image.shape[1]//4
            self.min_n = 2
            self.cliplimit = cliplimit
            logger.info("Object created successfully")
        except Exception as e:
            logger.error("Unexpected error occured while initialising the object: %s", e)
            raise
        
        
        
    
    def gen_population(self, chromosome_length, population_size):
        """Generate a population of the desired size"""
        try:
            population=[]
            for i in range(population_size):
                chromosome = random.choices([0, 1], k=chromosome_length)
                population.append(chromosome)

            return population
        except Exception as e:
            logger.error("Unexpected error occured while generating populatione: %s", e)
            raise    
    
    
    def bin2dec(self, chromosomes):
        """Convert the decimal values of height and width to decimal to know number of chromosomes"""
        try:
            decimal=0

            for chromosome in chromosomes:
                decimal=decimal*2 + int(chromosome)

            return decimal
        except Exception as e:
            logger.error("Unexpected error occured while converting height and width to their decimal representation: %s", e)
            raise    
    
    
    def divide_chromosome(self, chromosome):
        """Segregates the chromosomes into m(height) chromosomes and n(width) chromosomes"""
        try:
            mbits=len(chromosome)//2
            nbits=len(chromosome)-mbits

            m=chromosome[:mbits]
            n=chromosome[nbits:]

            m = self.bin2dec(m)
            n = self.bin2dec(n)

            return m,n
        except Exception as e:
            logger.error("Unexpected error occured while dividing the chromosome: %s", e)
            raise

    
    def calculate_fitness(self, enhanced_image):
        """Calculates fitness for enhanced value"""
        try:
            org_img=np.asarray(self.image)
            enh_img=np.asarray(enhanced_image)
            if len(enh_img.shape) == 3:
                enh_img = cv2.cvtColor(enh_img, cv2.COLOR_BGR2GRAY)

    # Convert original to grayscale if needed
            if len(org_img.shape) == 3:
                org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)

            mse=np.mean((org_img - enh_img) ** 2)
            maxpixel=255.0

            psnr=20*np.log10(maxpixel/np.sqrt(mse))

            return psnr
        except Exception as e:
            logger.error("Unexpected error occured while calculating fitness: %s", e)
            raise  
  
    
    def ApplyClahe(self, tilesize: tuple):
        """Apply CLAHE"""
        try:
            if len(self.image.shape) == 2:
                original_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            else:
                original_image = self.image


            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2LAB)            
            L,A,B=cv2.split(original_image)
            L = L.astype(np.uint8)
            
            CLAHE=cv2.createCLAHE(self.cliplimit,tilesize)
            Enh_L=CLAHE.apply(L)
            
            Enh_LAB=cv2.merge((Enh_L,A,B))

            return cv2.cvtColor(Enh_LAB,cv2.COLOR_LAB2BGR)
        except Exception as e:
            logger.error("Unexpected error occured while apply enhancement to the image: %s", e)
            raise    
    
    
    def generate_fitness(self, population, eachgentable):
        """Generate Fitness Values For Each Chromosomes"""
        try:
            Fitness_table = np.zeros(len(population))
            # print("Length Of Chromosome: ",len(population[0]))
            for i, current_chromosome in enumerate(population):
                m,n = self.divide_chromosome(current_chromosome)
                grid=(m,n)
                Enh_image = self.ApplyClahe(grid)
                Fitness_table[i] = self.calculate_fitness(Enh_image)
            eachgentable.append(Fitness_table)
            # plt.plot(Fitness_table)
            return Fitness_table, eachgentable
        except Exception as e:
            logger.error("Unexpected error occured while generating fitness table: %s", e)
            raise    
    
    
    def sortfitness(self, population, fitness_table, bestfitnesstable):
        """Sort The Fitness Values Generated In The generate_fitness function"""
        try:
            combined_arrays = [(population[i], fitness_table[i]) for i in range(len(population))]


            sorted_combined_arrays = sorted(combined_arrays, key=lambda x: x[1])

            sorted_population = [t[0] for t in sorted_combined_arrays]
            sorted_fitness_table = [t[1] for t in sorted_combined_arrays]

            # print("Fittest Chromosome's Score: ", sorted_fitness_table[len(population)-1], " Fittest Chromosome: ",sorted_population[len(population)-1])
            bestfitnesstable.append(sorted_fitness_table[len(population)-1])
            return sorted_population,sorted_fitness_table,bestfitnesstable
        except Exception as e:
            logger.error("Unexpected error occured while sorting fitness table: %s", e)
            raise    
    
    
    def roulette_wheel_selection(self, fitness_table, population):
        """Selection by roulette wheel"""
        try:
            fitness_cdf = np.cumsum(fitness_table)
            fitness_cdf = fitness_cdf / fitness_cdf[-1]

            rand_num = np.random.rand()
            for i in range(len(population)):
                if rand_num < fitness_cdf[i]:
                    selected_parent = population[i]

            return selected_parent
        except Exception as e:
            logger.error("Unexpected error occured while selecting parents for mutation: %s", e)
            raise


    def crossover(self, parent1,parent2):
        """Performing single parent crossover for variation"""
        try:
            point1,point2=np.random.choice(len(parent1),size=2,replace=False)
            child1=parent1[:min(point1,point2)]+parent2[min(point1,point2):max(point1,point2)]+parent1[max(point1,point2):]
            child2=parent2[:min(point1,point2)]+parent1[min(point1,point2):max(point1,point2)]+parent2[max(point1,point2):]
            return child1, child2
        except Exception as e:
            logger.error("Unexpected error occured while crossover: %s", e)
            raise  
    
    
    def newgeneration(self, child1,child2,population):
        """Generating new generation"""
        try:
            population[0]=child1
            population[1]=child2
            return population
        except Exception as e:
            logger.error("Unexpected error occured while generating the new generation: %s", e)
            raise    
    
    
    def mutation(self, population):
        """Introducing mutations"""
        try:
            selected_chromosome=random.randint(0,len(population)-2)

            point1=np.random.randint(0,len(population[0])-1)
            point2=np.random.randint(0,len(population[0])-1)
            point3=np.random.randint(0,len(population[0])-1)
            if population[selected_chromosome][point1]==1:
                population[selected_chromosome][point1]=0
            else:
                population[selected_chromosome][point1]=1

            if population[selected_chromosome][point2]==1:
                population[selected_chromosome][point2]=0
            else:
                population[selected_chromosome][point2]=1

            if population[selected_chromosome][point3]==1:
                population[selected_chromosome][point3]=0
            else:
                population[selected_chromosome][point3]=1

            return population
        except Exception as e:
            logger.error("Unexpected error occured while running mutating: %s", e)
            raise    
    
    
    def GA_for_CLAHE(self, population_size, generations):
        """Main function for implementing GA"""
        try:
            chromosome_length=len(bin(max(self.max_m,self.max_n))[2:])*2
            population=self.gen_population(chromosome_length,population_size)
            Best_fitness=[]
            eachgentable=[]

            # print("--------------------------------------------------------------------")

            for i in range(generations):
                # print("Generation: ", i+1)
                fitness_table,eachgentable=self.generate_fitness(population,eachgentable)
                population,fitness_table,Best_fitness=self.sortfitness(population,fitness_table,Best_fitness)

                parent1=self.roulette_wheel_selection(fitness_table,population)
                parent2=self.roulette_wheel_selection(fitness_table,population)
                child1,child2=self.crossover(parent1,parent2)
                population=self.newgeneration(child1,child2,population)

                parent1=self.roulette_wheel_selection(fitness_table,population)
                parent2=self.roulette_wheel_selection(fitness_table,population)
                child1,child2=self.crossover(parent1,parent2)
                population=self.newgeneration(child1,child2,population)

                parent1=self.roulette_wheel_selection(fitness_table,population)
                parent2=self.roulette_wheel_selection(fitness_table,population)
                child1,child2=self.crossover(parent1,parent2)
                population=self.newgeneration(child1,child2,population)

                population=self.mutation(population)
                population=self.mutation(population)
                population=self.mutation(population)
                population=self.mutation(population)


                # print("------------------------------------------------------------------")

            self.enhanced=self.applyclahe(population=population,population_size=population_size)
            logger.info("Best chromosome found: %s", population[population_size-1]) 

            return Best_fitness, eachgentable
        except Exception as e:
            logger.error("Unexpected error occured while running genetic algorithm: %s", e)
            raise    
    
    
    def applyclahe(self, population, population_size):
        """Function to apply CLAHE with the selected best chromosome"""
        try:
            best_chromosome = population[population_size-1]
            m,n = self.divide_chromosome(best_chromosome)

            m=max(min(m,self.max_m), 2)
            n=max(min(n,self.max_n), 2)

            # print(m)
            # print(n)
            if m>=1 and n>=1:
                grid=(m,n)
                self.tilesize = grid
                enhanced_image= self.ApplyClahe(self.tilesize)
            
            else: 
                enhanced_image= self.image 
            return enhanced_image
        except Exception as e:
            logger.error("Unexpected error occured while applying final clahe with best grid values: %s", e)
            raise    
    
    
    def RunGA(self, population_size = 20, generations = 50) -> np.array:
        """Code to run GA"""
        try:
            logger
            self.Best_fitness, self.eachgentable= self.GA_for_CLAHE(population_size, generations)
            return self.enhanced
        except Exception as e:
            logger.error("Unexpected error occured while inititating genetic algorithm: %s", e)
            raise
    
    
    def compare_images(self, enhanced_image):
        """Compare the enhanced and original image"""
        try:
            original_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)

            
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            
            axes[0].imshow(original_image)
            axes[0].set_title('Original')
            axes[0].axis('off')

            
            axes[1].imshow(enhanced_image)
            axes[1].set_title('Enhanced (CLAHE)')
            axes[1].axis('off')

            plt.savefig("result/Comparison.jpg")
        except Exception as e:
            logger.log("Unexpected error occured while saving the comparison image: %s", e)
            raise        
        
        
        
    