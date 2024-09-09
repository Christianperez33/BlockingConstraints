############################################################
#                       LIBRERIAS                          #
############################################################
import numpy as np
from scipy.stats import norm, uniform, expon
import pickle
import json
import copy
import pandas as pd
############################################################
#                       GENERADOR                          #
############################################################


def f(x):
    return int(np.exp(-int(x)/100)*100)

def g(x):
    return 90*x + 10

def t(c):
    return 4.0704 * np.log(2) / np.log(1 + (c* 2.5093)**3)


class JSP:
    def __init__(self, jobs, machines, ProcessingTime=[], EnergyConsumption=[], ReleaseDateDueDate=[], Orden=[]) -> None:       
        # Atributos del problema
        self.numJobs = jobs
        self.numMchs = machines

        self.speed             = ProcessingTime.shape[-1] if ProcessingTime else 0
        self.ProcessingTime    = ProcessingTime
        self.EnergyConsumption = EnergyConsumption
        self.Orden             = Orden
        self.rddd              = ReleaseDateDueDate.ndim - 1 if ReleaseDateDueDate else 0
        
    def fill_random_values(self, speed, rddd, distribution, seed):
        # Seteo de la semilla
        np.random.seed(seed)

        self.rddd = rddd
        self.speed = speed
        
        # Particion del intervalo [0.5, 3]
        energyPer, timePer = self._particionate_speed_space(speed)

        # Generacion de la matriz de costes para velocidad standar de cada operacion
        self._generate_standar_operation_cost(distribution)

        # Matriz que almacena el tiempo de procesamiento de cada trabajo en cada máquina
        self.ProcessingTime = np.zeros((self.numJobs, self.numMchs, self.speed), dtype=int)

        # Matriz que almacena el consumo de procesamiento de cada trabajo en cada máquina
        self.EnergyConsumption = np.zeros((self.numJobs, self.numMchs, self.speed), dtype=int)
    
        # Matriz que almacena el orden en el que se deben procesar las operaciones de cada job
        self.Orden  = np.zeros((self.numJobs, self.numMchs),dtype=int)

        # Matriz que almacena la fecha de disponibilidad de cada trabajo
        if self.rddd == 0:
            release_date_tasks = np.array([0]*self.numJobs)
        
        elif self.rddd == 1:
            release_date_tasks = np.random.choice(range(0,101,10), self.numJobs)
            release_date_tasks = release_date_tasks - release_date_tasks.min()
            # Matriz encargada de almacenar el release date y due date de los trabajos/operaciones
            self.ReleaseDueDate = np.zeros((self.numJobs, 2), dtype=int)

        elif self.rddd == 2 :
            release_date_tasks = np.random.choice(range(0,101,10), self.numJobs)
            release_date_tasks = release_date_tasks - release_date_tasks.min()
            # En caso de que sea a nivel de operaciones
            self.ReleaseDueDate = np.zeros((self.numJobs,self.numMchs,2), dtype=int)

        self._jobToMachine(release_date_tasks, timePer, distribution)

    def _particionate_speed_space(self, speed):
        energyPer    = np.linspace(0.5,3,speed) if speed > 1 else [1]
        timePer      = [t(c) for c in energyPer]
        return energyPer, timePer

    def _generate_standar_operation_cost(self, distribution):
        if distribution == "uniform":
            self.operationCost = np.random.uniform(10,100,(self.numJobs, self.numMchs))
        elif distribution == "normal":
            self.operationCost = np.array([max(10,x) for x in np.random.normal(50,20,(self.numJobs, self.numMchs)).reshape(-1)]).reshape(self.numJobs,self.numMchs)
        elif distribution == "exponential":
            self.operationCost = np.array([max(10,x) for x in np.random.exponential(50,(self.numJobs, self.numMchs)).reshape(-1)]).reshape(self.numJobs,self.numMchs)
    
    def _jobToMachine(self,release_date_tasks, timePer, distribution):
        for job in range(self.numJobs):
            machines        = np.random.choice(range(self.numMchs), self.numMchs, replace=False)
            self.Orden[job] = machines
            releaseDateTask = release_date_tasks[job]
            initial = releaseDateTask
            for machine in machines:
                for S, (proc,energy) in enumerate(self._genProcEnergy(job, machine, timePer)):
                    self.ProcessingTime[job,machine,S] = proc
                    self.EnergyConsumption[job,machine,S] = energy
                if self.rddd == 2:
                    self.ReleaseDueDate[job,machine,0] = releaseDateTask
                    releaseDateTask += int(self._release_due(np.median(self.ProcessingTime[job,machine,:]), distribution))
                    self.ReleaseDueDate[job,machine,1] = releaseDateTask
                else:
                    releaseDateTask += np.median(self.ProcessingTime[job,machine,:])
            if self.rddd == 1:
                self.ReleaseDueDate[job] = [initial, int(self._release_due(releaseDateTask, distribution))]

    # Método encargado de generar el tiempo que tarda en ser procesado un trabajo y su consumo, teniendo en cuenta las
    # distintas velocidades
    def _genProcEnergy(self, job, machine, timePer):        
        ans = []  
        for tper in timePer:
            time = max(1,self.operationCost[job,machine] * tper)
            ans.append((time, max(1,f(time))))
        return ans

    def _release_due(self, duration, distribution):
        if distribution == "uniform":
            return uniform(duration,2*duration).rvs()
        elif distribution == "normal":
            return max(duration,norm(loc=2*duration, scale=duration/2).rvs())
        else:
            return max(duration,expon(loc=duration,scale=duration/2).rvs())

    def savePythonFile(self,path):
        with open(path,'wb') as f:
            pickle.dump(self,f)
    
    def saveJsonFile(self, path):
        self.JSP = {
                "nbJobs":list(range(self.numJobs)),
                "nbMchs": list(range(self.numMchs)),
                "speed" : self.speed,
                "timeEnergy":[],
                "minMakespan": int(self.min_makespan),
                "minEnergy": int(self.min_energy),
                "maxMinMakespan" : int(self.max_min_makespan),
                "maxMinEnergy" : int(self.max_min_energy)
            }
        
        for job in range(self.numJobs):
            new = {
                "jobId" : job,
                "operations":{}
            }
            for machine in self.Orden[job]:
                machine = int(machine)
                new["operations"][machine] = {"speed-scaling" : 
                                              [
                                                {"procTime" : int(proc),
                                                 "energyCons" : int(energy)
                                                }
                                                for proc,energy in zip(self.ProcessingTime[job, machine],self.EnergyConsumption[job, machine])
                                              ]
                                              }
                if self.rddd == 2:
                    new["operations"][machine]["release-date"] = int(self.ReleaseDueDate[job][machine][0])
                    new["operations"][machine]["due-date"]     = int(self.ReleaseDueDate[job][machine][1])
            if self.rddd == 1:
                # A nivel de job
                new["release-date"] = int(self.ReleaseDueDate[job][0])
                new["due-date"] = int(self.ReleaseDueDate[job][1])
            if self.rddd == 2:
                new["release-date"] = int(min(self.ReleaseDueDate[job,:,0]))
                new["due-date"]     = int(max(self.ReleaseDueDate[job,:,1]))
            self.JSP["timeEnergy"].append(new)

        with open(path, 'w') as f:
            json.dump(self.JSP, f,indent=4)

    def select_speeds(self, speeds):
        if self.speed == len(speeds):
            return self
        # Copiamos el objeto 
        new_object = copy.deepcopy(self)
        # Actualizar las velocidades del problema
        new_object.speed = len(speeds)
        
        # Actualizamos las matrices de velocidades y de consumi con las nuevas velocidades
        new_object.ProcessingTime = new_object.ProcessingTime[:,:,speeds]
        new_object.EnergyConsumption = new_object.EnergyConsumption[:,:,speeds]

        # Actualizamos los valores maximos y minimos de las funciones objetivo
        new_object.generate_maxmin_objective_values()
        return new_object

    def change_rddd_type(self, new_rddd):
        # En caso de que el tipo que se desee sea el mismo que ya se tiene, no se realiza ningún cambio
        if new_rddd == self.rddd:
            return self
        # En caso de que sea distinto se procede a cambiar los datos
        # Copiamos el objeto 
        new_object = copy.deepcopy(self)
        # Actualizamos el valor de rddd
        new_object.rddd = new_rddd 
        # Actualizamos la matriz de `release date and due date`
        if new_rddd == 0:
            if self.rddd != 0:
                del new_object.ReleaseDueDate
        elif new_rddd == 1:
            if self.rddd == 0:
                pass
            elif self.rddd == 1:
                pass
            elif self.rddd == 2:
                new_object.ReleaseDueDate = np.zeros((self.numJobs, 2), dtype=int)
                for job in range(self.numJobs):
                    new_object.ReleaseDueDate[job] = min(self.ReleaseDueDate[job,:,0]), max(self.ReleaseDueDate[job,:,1])
        elif new_rddd == 2:
            pass
        # Actualizamos los valores maximos y minimos de las funciones objetivo
        new_object.generate_maxmin_objective_values()
        return new_object

    # Métodos utilizados en la resolución con metaheuristicas
    def generate_maxmin_objective_values(self):
        # Makespan
        max_makespan           = sum([max(self.ProcessingTime[job,machine,:]) for job in range(self.numJobs) for machine in range(self.numMchs)])
        self.min_makespan      = max([sum([ min(self.ProcessingTime[job,machine,:]) for machine in range(self.numMchs)]) for job in range(self.numJobs)])
        self.max_makespan      = max_makespan
        self.max_min_makespan  = max_makespan - self.min_makespan
        # Energy
        max_energy          = sum([max(self.EnergyConsumption[job,machine,:]) for job in range(self.numJobs) for machine in range(self.numMchs)])
        self.min_energy     = sum([min(self.EnergyConsumption[job,machine,:]) for job in range(self.numJobs) for machine in range(self.numMchs)])
        self.max_energy     = max_energy
        self.max_min_energy = max_energy - self.min_energy
        # Tardiness
        if self.rddd > 0:
            self.max_tardiness = self.max_makespan
        else:
            self.max_tardiness = -1

    def norm_makespan(self,makespan):
        return (makespan - self.min_makespan) / self.max_min_makespan
    
    def norm_energy(self, energy):
        return (energy - self.min_energy) / self.max_min_energy if self.max_min_energy > 0 else 0
    
    def norm_tardiness(self, tardiness):
        return tardiness / self.max_tardiness if self.rddd > 0 else 0
    
    def objective_function_solution(self, solution):
        makespan  = 0
        energy    = 0
        tardiness = 0
        
        # Datos que se necesitan para calcular el makespan
        orders_done              = [0] * self.numJobs
        available_time_machines  = [0] * self.numMchs
        end_time_last_operations = [0] * self.numJobs
        
        tproc = [0]*self.numJobs
        for job,speed in zip(solution[::2], solution[1::2]):
            operation    = orders_done[job]
            machine      = self.Orden[job,operation]            
            
            # Tiempo en el que ha terminado la ultima operacion del trabajo
            end_time_last_operation = end_time_last_operations[job]
            
            # Tiempo de disponibilidad de la maquina
            available_time          = available_time_machines[machine]
            
            if operation == 0:
                if self.rddd == 0:
                    release_date = 0
                elif self.rddd == 1:
                    release_date = self.ReleaseDueDate[job,0]
                elif self.rddd == 2:
                    release_date = self.ReleaseDueDate[job,machine,0]
            else:                
                if self.rddd == 2:
                    release_date = self.ReleaseDueDate[job,machine,0]
                else:
                    release_date = available_time

            start_time = max(end_time_last_operation, available_time, release_date)
            end_time   = start_time + self.ProcessingTime[job, machine,speed]

            if self.rddd == 2:
                tardiness +=  min(max(0, end_time - self.ReleaseDueDate[job,machine,1]), self.ProcessingTime[job,machine, speed])
            energy += self.EnergyConsumption[job,machine, speed]
            if self.rddd == 1:
                tproc[job] += self.ProcessingTime[job, machine, speed]
            # Actualizamos la matriz
            available_time_machines[machine] = end_time
            end_time_last_operations[job]    = end_time
            orders_done[job] += 1
            # print(orders_done)
            # print(available_time_machines)
            # print(end_time_last_operations)
        
        makespan = max(end_time_last_operations)

        if self.rddd == 1:
            tardiness = sum(min(max(0, end_time - self.ReleaseDueDate[job,1] ), tproc[job]) for job, end_time in enumerate(end_time_last_operations))

        return self.norm_makespan(makespan) + self.norm_energy(energy) + self.norm_tardiness(tardiness), (makespan, energy, tardiness)
    
    def evalua_añadir_operacion(self, candidate, speed, makespan, energy, tardiness, orders_done, available_time_machines, end_time_last_operations,tproc, actualizacion):
        operation    = orders_done[candidate]
        machine      = self.Orden[candidate,operation]            
        
        # Tiempo en el que ha terminado la ultima operacion del trabajo
        end_time_last_operation = end_time_last_operations[candidate]
        
        # Tiempo de disponibilidad de la maquina
        available_time          = available_time_machines[machine]
        
        if operation == 0:
            if self.rddd == 0:
                release_date = 0
            elif self.rddd == 1:
                release_date = self.ReleaseDueDate[candidate,0]
            elif self.rddd == 2:
                release_date = self.ReleaseDueDate[candidate,machine,0]
        else:                
            if self.rddd == 2:
                release_date = self.ReleaseDueDate[candidate,machine,0]
            else:
                release_date = available_time

        start_time = max(end_time_last_operation, available_time, release_date)
        end_time   = start_time + self.ProcessingTime[candidate, machine,speed]

        if self.rddd == 2:
            tardiness += min(max(0, end_time - self.ReleaseDueDate[candidate,machine,1]), self.ProcessingTime[candidate, machine, speed])
        energy += self.EnergyConsumption[candidate,machine, speed]

        if actualizacion:
            # Actualizamos la matriz
            available_time_machines[machine] = end_time
            end_time_last_operations[candidate]    = end_time
            orders_done[candidate] += 1
            tproc[candidate]              += self.ProcessingTime[candidate, machine, speed]

        makespan = makespan if end_time < makespan else end_time
        
        if self.rddd == 1:
            tardiness = sum(min(max(0, end_time - self.ReleaseDueDate[job,1]), tproc[job]) for job, end_time in enumerate(end_time_last_operations))

        return self.norm_makespan(makespan) + self.norm_energy(energy) + self.norm_tardiness(tardiness), makespan, energy, tardiness
    
    def generate_schedule_image(self, schedule):
        pass

    
    ######################
    ##     FEATURES     ##
    ######################
    def std_makespan(self):
        norm_time = (self.ProcessingTime - self.ProcessingTime.min())/(self.ProcessingTime.max() - self.ProcessingTime.min())

        std_job = [np.std(norm_time[job,:,:]) for job in range(self.numJobs)]
        std_task = [np.std(norm_time[:,machine,:]) for machine in range(self.numMchs)]
        
        
        return std_job,std_task
    
    def std_energy(self):
        norm_energy = (self.EnergyConsumption - self.EnergyConsumption.min())/(self.EnergyConsumption.max() - self.EnergyConsumption.min())

        std_job = [np.std(norm_energy[job,:,:]) for job in range(self.numJobs)]
        std_task = [np.std(norm_energy[:,machine,:]) for machine in range(self.numMchs)]
        
        
        return std_job,std_task
    
    def features(self):
        features = {}
        # Basic features
        features["jobs"]           = self.numJobs
        features["machines"]       = self.numMchs
        features["rddd"]           = self.rddd
        features["speed"]          = self.speed
        
        features["max_makespan"]    = self.max_makespan
        features["min_makespan"]    = self.min_makespan
        features["max_sum_energy"]   = self.max_energy
        features["min_sum_energy"]   = self.min_energy
        features["max_tardiness"]   = self.max_tardiness
        features["min_window"]     = 0
        features["max_window"]     = 0
        features["mean_window"]    = 0
        features["overlap"]        = 0
        ##New features
        # Makespan
        features["max_makespan_job"]   = [self.ProcessingTime[job,:,:].max() for job in range(self.numJobs)]
        features["max_makespan_task"]  = [self.ProcessingTime[:,machine,:].max() for machine in range(self.numMchs)]
        features["min_makespan_job"]   = [self.ProcessingTime[job,:,:].min() for job in range(self.numJobs)]
        features["min_makespan_task"]  = [self.ProcessingTime[:,machine,:].min() for machine in range(self.numMchs)]
        features["std_makespan_job"],features["std_makespan_task"] = self.std_makespan()
        # Energy
        features["max_energy_job"]     = [self.EnergyConsumption[job,:,:].max() for job in range(self.numJobs)]
        features["max_energy_task"]    = [self.EnergyConsumption[:,machine,:].max() for machine in range(self.numMchs)]
        features["min_energy_job"]     = [self.EnergyConsumption[job,:,:].min() for job in range(self.numJobs)]
        features["min_energy_task"]    = [self.EnergyConsumption[:,machine,:].min() for machine in range(self.numMchs)]
        features["std_energy_job"],features["std_energy_task"] = self.std_energy()
        # Tardiness
        features["tardiness"] = np.zeros((self.numJobs,self.numMchs))
        # if self.rddd > 0:
        #     features["max_tardiness"] = self.max_makespan
        # else:
        
        
        if self.rddd == 0:
            features["min_window"]  = -1
            features["max_window"]  = -1
            features["mean_window"]  = -1
            features["overlap"] = -1
        else:
            if self.rddd == 1:
                # Ventana de cada trabajo
                for job in range(self.numJobs):
                    tproc_min  = np.sum(np.min(self.ProcessingTime[job,machine,:]) for machine in range(self.numMchs))
                    tproc_max  = np.sum(np.max(self.ProcessingTime[job,machine,:]) for machine in range(self.numMchs))                    
                    tproc_mean = np.sum(np.mean(self.ProcessingTime[job,machine,:]) for machine in range(self.numMchs)) 
                    window     = self.ReleaseDueDate[job,1] - self.ReleaseDueDate[job,0]
                    features["min_window"]  += window / tproc_max 
                    features["max_window"]  += window / tproc_min 
                    features["mean_window"] += window / tproc_mean 
                features["min_window"]  = features["min_window"]  / self.numJobs
                features["max_window"]  = features["max_window"]  / self.numJobs
                features["mean_window"] = features["mean_window"] / self.numJobs
                # Overlap entre trabajos
                for job in range(self.numJobs):
                    for job2 in range(job + 1, self.numJobs):
                        diff = min(self.ReleaseDueDate[job,1],self.ReleaseDueDate[job2,1])-max(self.ReleaseDueDate[job,0], self.ReleaseDueDate[job2,0])
                        if diff > 0:
                            features["overlap"] += diff / (self.ReleaseDueDate[job,1] - self.ReleaseDueDate[job,0])
                            features["overlap"] += diff / (self.ReleaseDueDate[job2,1] - self.ReleaseDueDate[job2,0])
                features["overlap"] = features["overlap"] / (self.numJobs * (self.numJobs - 1))
            else:
                # Ventana de cada operacion
                for job in range(self.numJobs):
                    for machine in range(self.numMchs):
                        tproc_min  = np.min(self.ProcessingTime[job,machine,:])
                        tproc_max  = np.max(self.ProcessingTime[job,machine,:])                   
                        tproc_mean = np.mean(self.ProcessingTime[job,machine,:])
                        window     = self.ReleaseDueDate[job,machine,1] - self.ReleaseDueDate[job,machine,0]
                        features["min_window"]  += window / tproc_max 
                        features["max_window"]  += window / tproc_min 
                        features["mean_window"] += window / tproc_mean 
                features["min_window"]  = features["min_window"]  / (self.numJobs * self.numMchs)
                features["max_window"]  = features["max_window"]  / (self.numJobs * self.numMchs)
                features["mean_window"] = features["mean_window"] / (self.numJobs * self.numMchs)
                # Overlap entre operaciones
                for job1 in range(self.numJobs):
                    for machine1 in range(self.numMchs):
                        for job2 in range(job1 + 1, self.numJobs):
                            diff = min(self.ReleaseDueDate[job1,machine1,1],self.ReleaseDueDate[job2,machine1,1])-max(self.ReleaseDueDate[job1,machine1,0],  self.ReleaseDueDate[job2,machine1,0])
                            if diff > 0:
                                features["overlap"] += diff / (self.ReleaseDueDate[job1,machine1,1] - self.ReleaseDueDate[job1,machine1,0])
                                features["overlap"] += diff / (self.ReleaseDueDate[job2,machine1,1] -  self.ReleaseDueDate[job2,machine1,0])
                features["overlap"] = features["overlap"] / (self.numJobs * (self.numJobs - 1) * self.numMchs)   
        # Estadísticos de los datos
        features["max_processing_time_value"]     = np.max(self.ProcessingTime)
        features["min_processing_time_value"]     = np.min(self.ProcessingTime)
        features["mean_processing_time_value"]    = np.mean(self.ProcessingTime)

        features["max_energy_value"]     = np.max(self.ProcessingTime)
        features["min_energy_value"]     = np.min(self.ProcessingTime)
        features["mean_energy_value"]    = np.mean(self.ProcessingTime)
        
        ##Correlación de Pearson - medida de dependencia lineal entre dos variables aleatorias cuantitativas
        #  R = 1 -> dependencia total entre las dos variables denominada relación directa: cuando una de ellas aumenta, la otra también lo hace en proporción constante.
        #  0 < R < 1 -> correlación positiva
        # -1 < R < 0 -> correlación negativa
        #  R = -1 -> dependencia total entre las dos variables llamada relación opuesta: cuando una de ellas aumenta, la otra cambia su signo en proporción constante.
        #Cuanto mayor sea R mayor mas rapido crecen los valores time-energy y viceversa
        features["pearson"]={"job":{},"machine":{},"speed":{},"job_machine":np.zeros((self.numJobs,self.numMchs)),"job_speed":np.zeros((self.numJobs,self.speed)),"machine_speed":np.zeros((self.numMchs,self.speed))}

        
        for j in range(self.numJobs):
            features["pearson"]["job"][j] = np.corrcoef(self.ProcessingTime[j],self.EnergyConsumption[j],rowvar=self.speed!=1).sum()
            for m in range(self.numMchs):
                features["pearson"]["machine"][j] = np.corrcoef(self.ProcessingTime[:,m],self.EnergyConsumption[:,m],rowvar=self.speed!=1).sum()
                if self.speed!=1:
                    features["pearson"]["job_machine"][j,m] = np.corrcoef(self.ProcessingTime[j,m,:],self.EnergyConsumption[j,m,:]).sum()
                for s in range(self.speed):
                    features["pearson"]["speed"][s] = np.corrcoef(self.ProcessingTime[:,:,s],self.EnergyConsumption[:,:,s],rowvar=self.speed!=1).sum()
                    features["pearson"]["job_speed"][j,s] = np.corrcoef(self.ProcessingTime[j,:,s],self.EnergyConsumption[j,:,s],rowvar=self.speed!=1).sum()
                    features["pearson"]["machine_speed"][m,s] = np.corrcoef(self.ProcessingTime[:,m,s],self.EnergyConsumption[:,m,s],rowvar=self.speed!=1).sum()
        
        print(features["pearson"])
        
        quit()
        return features
