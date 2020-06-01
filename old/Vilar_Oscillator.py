#!/usr/bin/env python
# coding: utf-8

# # Vilar_Oscillator

# In[2]:


import numpy as np
import gillespy2
from gillespy2.core import Model, Species, Reaction, Parameter, RateRule


# In[3]:


class Vilar_Oscillator(Model):
    def __init__(self, parameter_values=None):
        Model.__init__(self, name="Vilar_Oscillator")
        self.volume = 1

        # Parameters
        self.add_parameter(Parameter(name="alpha_a", expression=50))
        self.add_parameter(Parameter(name="alpha_a_prime", expression=500))
        self.add_parameter(Parameter(name="alpha_r", expression=0.01))
        self.add_parameter(Parameter(name="alpha_r_prime", expression=50))
        self.add_parameter(Parameter(name="beta_a", expression=50))
        self.add_parameter(Parameter(name="beta_r", expression=5))
        self.add_parameter(Parameter(name="delta_ma", expression=10))
        self.add_parameter(Parameter(name="delta_mr", expression=0.5))
        self.add_parameter(Parameter(name="delta_a", expression=1))
        self.add_parameter(Parameter(name="delta_r", expression=0.2))
        self.add_parameter(Parameter(name="gamma_a", expression=1))
        self.add_parameter(Parameter(name="gamma_r", expression=1))
        self.add_parameter(Parameter(name="gamma_c", expression=2))
        self.add_parameter(Parameter(name="theta_a", expression=50))
        self.add_parameter(Parameter(name="theta_r", expression=100))

        # Species
        self.add_species(Species(name="Da", initial_value=1, mode="dynamic"))
        self.add_species(Species(name="Da_prime", initial_value=0, mode="dynamic"))
        self.add_species(Species(name="Ma", initial_value=0, mode="dynamic"))
        self.add_species(Species(name="Dr", initial_value=1, mode="dynamic"))
        self.add_species(Species(name="Dr_prime", initial_value=0, mode="dynamic"))
        self.add_species(Species(name="Mr", initial_value=0, mode="dynamic"))
        self.add_species(Species(name="C", initial_value=10, mode="dynamic"))
        self.add_species(Species(name="A", initial_value=10, mode="dynamic"))
        self.add_species(Species(name="R", initial_value=10, mode="dynamic"))

        # Reactions
        self.add_reaction(Reaction(name="r1", reactants={'Da_prime': 1}, products={'Da': 1}, rate=self.listOfParameters["theta_a"]))
        self.add_reaction(Reaction(name="r2", reactants={'Da': 1, 'A': 1}, products={'Da_prime': 1}, rate=self.listOfParameters["gamma_a"]))
        self.add_reaction(Reaction(name="r3", reactants={'Dr_prime': 1}, products={'Dr': 1}, rate=self.listOfParameters["theta_r"]))
        self.add_reaction(Reaction(name="r4", reactants={'Dr': 1, 'A': 1}, products={'Dr_prime': 1}, rate=self.listOfParameters["gamma_r"]))
        self.add_reaction(Reaction(name="r5", reactants={'Da_prime': 1}, products={'Da_prime': 1, 'Ma': 1}, rate=self.listOfParameters["alpha_a_prime"]))
        self.add_reaction(Reaction(name="r6", reactants={'Da': 1}, products={'Da': 1, 'Ma': 1}, rate=self.listOfParameters["alpha_a"]))
        self.add_reaction(Reaction(name="r7", reactants={'Ma': 1}, products={}, rate=self.listOfParameters["delta_ma"]))
        self.add_reaction(Reaction(name="r8", reactants={'Ma': 1}, products={'A': 1, 'Ma': 1}, rate=self.listOfParameters["beta_a"]))
        self.add_reaction(Reaction(name="r9", reactants={'Da_prime': 1}, products={'Da_prime': 1, 'A': 1}, rate=self.listOfParameters["theta_a"]))
        self.add_reaction(Reaction(name="r10", reactants={'Dr_prime': 1}, products={'Dr_prime': 1, 'A': 1}, rate=self.listOfParameters["theta_a"]))
        self.add_reaction(Reaction(name="r11", reactants={'A': 1}, products={}, rate=self.listOfParameters["gamma_c"]))
        self.add_reaction(Reaction(name="r12", reactants={'A': 1, 'R': 1}, products={'C': 1}, rate=self.listOfParameters["gamma_c"]))
        self.add_reaction(Reaction(name="r13", reactants={'Dr_prime': 1}, products={'Dr_prime': 1, 'Mr': 1}, rate=self.listOfParameters["alpha_r_prime"]))
        self.add_reaction(Reaction(name="r14", reactants={'Dr': 1}, products={'Dr': 1, 'Mr': 1}, rate=self.listOfParameters["alpha_r"]))
        self.add_reaction(Reaction(name="r15", reactants={'Mr': 1}, products={}, rate=self.listOfParameters["delta_mr"]))
        self.add_reaction(Reaction(name="r16", reactants={'Mr': 1}, products={'Mr': 1, 'R': 1}, rate=self.listOfParameters["beta_r"]))
        self.add_reaction(Reaction(name="r17", reactants={'R': 1}, products={}, rate=self.listOfParameters["delta_r"]))
        self.add_reaction(Reaction(name="r18", reactants={'C': 1}, products={'R': 1}, rate=self.listOfParameters["delta_a"]))

        # Timespan
        self.timespan(np.linspace(0, 200, 201))


# In[4]:


model = Vilar_Oscillator()


# In[5]:


results = model.run(seed=None)


# In[6]:


results.plotplotly()


# In[7]:


results


# In[ ]:




