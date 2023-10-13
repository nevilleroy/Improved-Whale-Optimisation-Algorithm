import numpy as np

class WhaleOptimization():

    def __init__(self, test_function, constraints, algorithm_mode, nsols, b, a, a_step, maximize=False):
        self.test_function = test_function
        self.constraints = constraints
        self.solutions = self._init_solutions(nsols) 
        self.b = b
        self.a = a
        self.a_step = a_step
        self.maximize = maximize
        self.best_solutions = []
        self.algorithm_mode = algorithm_mode
        self.attack_coefficient = self._calculate_attack_coefficient()

    def print_all_members(self):
        print (f"opt_func: {self.test_function}, constraints: {self.constraints}")
        print (f"b: {self.b}")
        print (f"a: {self.a}, a_step: {self.a_step}")
        print (f"maximize: {self.maximize}")
        print (f"Attack Coefficient: ", self.attack_coefficient)
        
    def get_solutions(self):
        """return solutions"""
        return self.solutions
                                                                  
    def optimize(self):
        """solutions randomly encircle, search or attack"""
        ranked_sol = self._rank_solutions()
        best_sol = ranked_sol[0] 
        
        #include best solution in next generation solutions
        new_sols = [best_sol]
                                                                 
        for s in ranked_sol[1:]:
            if self.is_attack_optimal(): 
                new_s = self._attack(s, best_sol)                                                     
            else:                                                                         
                new_s = self._attack(s, best_sol) 
                A = self._compute_A()                                                     
                norm_A = np.linalg.norm(A)                                                
                if norm_A < 1.5:                                                          
                    new_s = self._encircle(s, best_sol, A)                                
                else:                                                                     
                    #select random sol                                                  
                    random_sol = self.solutions[np.random.randint(self.solutions.shape[0])]       
                    new_s = self._search(s, random_sol, A)  

            new_sols.append(self._constrain_solution(new_s))
        self.solutions = np.stack(new_sols)
        self.a -= self.a_step

    def _init_solutions(self, nsols):
        """initialize solutions uniform randomly in space"""
        sols = []
        for c in self.constraints:
            sols.append(np.random.uniform(c[0], c[1], size=nsols))
                                                                            
        sols = np.stack(sols, axis=-1)
        return sols

    def _constrain_solution(self, sol):
        """ensure solutions are valid wrt to constraints"""
        constrain_s = []
        for c, s in zip(self.constraints, sol):
            if c[0] > s:
                s = c[0]
            elif c[1] < s:
                s = c[1]
            constrain_s.append(s)
        return constrain_s

    def _rank_solutions(self):
        """find best solution"""
        fitness = self.test_function(self.solutions[:, 0], self.solutions[:, 1])
        sol_fitness = [(f, s) for f, s in zip(fitness, self.solutions)]
   
        #best solution is at the front of the list
        ranked_sol = list(sorted(sol_fitness, key=lambda x:x[0], reverse=self.maximize))
        self.best_solutions.append(ranked_sol[0])

        return [s[1] for s in ranked_sol]

    def print_best_solutions(self):
        print('generation best solution history')
        print('([fitness], [solution])')
        for s in self.best_solutions:
            print(s)
        print('\n')
        print('best solution')
        print('([fitness], [solution])')
        print(sorted(self.best_solutions, key=lambda x:x[0], reverse=self.maximize)[0])

    def _compute_A(self):
        r = np.random.uniform(0.0, 1.0, size=2)
        return (2.0*np.multiply(self.a, r))-self.a

    def _compute_C(self):
        return 2.0*np.random.uniform(0.0, 1.0, size=2)
                                                                 
    def _encircle(self, sol, best_sol, A):
        D = self._encircle_D(sol, best_sol)
        return best_sol - np.multiply(A, D)
                                                                 
    def _encircle_D(self, sol, best_sol):
        C = self._compute_C()
        D = np.linalg.norm(np.multiply(C, best_sol)  - sol)
        return D

    def _calculate_attack_coefficient(self):
        if self.algorithm_mode == "original":
            print ("Mode Original!")
            return 1.0
        else:
            return 0.3

    def _search(self, sol, rand_sol, A):
        D = self._search_D(sol, rand_sol)
        return rand_sol - np.multiply(A, D)

    def _search_D(self, sol, rand_sol):
        C = self._compute_C()
        return np.linalg.norm(np.multiply(C, rand_sol) - sol)    

    def _attack(self, sol, best_sol):
        D = np.linalg.norm(best_sol - sol)
        L = np.random.uniform(-1.0, 1.0, size=2)
        return np.multiply(np.multiply(D,np.exp(self.b*L)), np.cos(2.0*np.pi*L))+best_sol

    def is_attack_optimal(self):
        if np.random.uniform(0.0, 1.0) > self.attack_coefficient:
            return False
        else:
            return True