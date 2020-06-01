#!/usr/bin/env python
# coding: utf-8

# # Vilar_Oscillator

# In[1]:


import numpy as np
import gillespy2
from gillespy2.core import Model, Species, Reaction, Parameter, RateRule, AssignmentRule, FunctionDefinition
from gillespy2.core.events import EventAssignment, EventTrigger, Event


# In[2]:


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
        self.add_species(Species(name="Da", initial_value=1, mode="discrete"))
        self.add_species(Species(name="Da_prime", initial_value=0, mode="discrete"))
        self.add_species(Species(name="Ma", initial_value=0, mode="discrete"))
        self.add_species(Species(name="Dr", initial_value=1, mode="discrete"))
        self.add_species(Species(name="Dr_prime", initial_value=0, mode="discrete"))
        self.add_species(Species(name="Mr", initial_value=0, mode="discrete"))
        self.add_species(Species(name="C", initial_value=10, mode="discrete"))
        self.add_species(Species(name="A", initial_value=10, mode="discrete"))
        self.add_species(Species(name="R", initial_value=10, mode="discrete"))

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


# In[3]:


model = Vilar_Oscillator()


# In[4]:


#from dask.utils import ensure_dict, format_bytes
import dask.utils
print(dask.__version__)
# fcn_list = [o[0] for o in getmembers(sys.modules[__name__], isfunction)]


# In[5]:


from tsfresh.feature_extraction.settings import MinimalFCParameters
from sciope.utilities.priors import uniform_prior
from sciope.utilities.summarystats import auto_tsfresh
from sciope.utilities.distancefunctions import naive_squared
from sciope.inference.abc_inference import ABC
from sklearn.metrics import mean_absolute_error
#from dask.distributed import Client


# In[6]:


# Define simulator function
def set_model_parameters(params, model):
    """para,s - array, need to have the same order as
    model.listOfParameters """
    for e, (pname, p) in enumerate(model.listOfParameters.items()):
        model.get_parameter(pname).set_expression(params[e])
    return model

# Here we use the GillesPy2 Solver
def simulator(params, model):
    model_update = set_model_parameters(params, model)
    num_trajectories = 1

    res = model_update.run(show_labels=False, number_of_trajectories=1, seed=None)
    tot_res = np.asarray([x.T for x in res]) # reshape to (N, S, T)
    tot_res = tot_res[:,1:, :] # should not contain timepoints

    return tot_res

# Wrapper, simulator function to abc should should only take one argument (the parameter point)
def simulator2(x):
    return simulator(x, model=model)


# In[7]:


# Set up the prior
default_param = np.array(list(model.listOfParameters.items()))[:,1] # take default from mode 1 as reference

bound = []
for exp in default_param:
    bound.append(float(exp.expression))

# Set the bounds
bound = np.array(bound)
dmin = bound * 0.1
dmax = bound * 2.0

# Here we use uniform prior
uni_prior = uniform_prior.UniformPrior(dmin, dmax)


# In[8]:


# generate some fixed(observed) data based on default parameters of the model
# the minimum number of trajectoies needs to be at least 30
fixed_data = model.run(show_labels=False, number_of_trajectories=100, seed=None)


# In[9]:


# Reshape the dat to (n_points,n_species,n_timepoints) and remove timepoints array
fixed_data = np.asarray([x.T for x in fixed_data])
fixed_data = fixed_data[:,1:, :]


# In[10]:


# Function to generate summary statistics
summ_func = auto_tsfresh.SummariesTSFRESH()

# Distance
ns = naive_squared.NaiveSquaredDistance()

# Start abc instance
abc = ABC(fixed_data, sim=simulator2, prior_function=uni_prior, summaries_function=summ_func.compute, distance_function=ns)


# In[11]:


# c = Client()
# c


# In[12]:


# First compute the fixed(observed) mean
# abc.compute_fixed_mean(chunk_size=2)


# In[13]:


# Run in multiprocessing mode
# res = abc.infer(num_samples=10, batch_size=10, chunk_size=2)


# In[14]:


# mae_inference = mean_absolute_error(bound, abc.results["inferred_parameters"])


# In[15]:


fixed_data[0]
fixed_data.shape 
# (100, 9, 201)
# > 100 trajectories, 9 species, 201 time points


# In[16]:


labels = -np.ones(fixed_data.shape[0])
# -1 = no label.
# 1 = :-)
# 2 = :-(


# In[24]:


print(labels)


# In[18]:



# As a user of the notebook, your API looks something like this:

i = -1
def get_id():
    # The ID (index) of the next plot to show to the user.
    # (TODO: this needs to be more like.. "whats the next one after ID X?" or "whats the next one without a label? 
    # -- this precludes pre-fetching
    global i
    i = i + 1
    return i



def on_new_label(id, label):
    # callback invoked when the user labels a plot.
    # needs to return 'quickly' to complete the browser request.
    labels[id] = label
    print(f"timeseries {id} has label {label}")
    
    
    
    


# In[20]:


import json



def do_get(handler):
    #print(handler.path)
    if handler.path == '/':    
        # Construct a server response.
        handler.send_response(200)
        handler.send_header('Content-type', 'text/html')
        handler.end_headers()
        
        f = open("active_learning.html", "r")
        handler.wfile.write(bytes("\n".join(f.readlines()), 'utf-8'))   
        f.close()

    elif handler.path == '/next_data':    
        # Construct a server response.
        handler.send_response(200)
        handler.send_header('Content-type', 'application/json')
        handler.end_headers()
        # just a dumb increment for now. This breaks REST, GETs need to have no side effects.
        # e.g. return next ID on post, then use get to get data only? 
        time_series_to_get = get_id()
        #print(f'getting {time_series_to_get}')
        handler.wfile.write(bytes(json.dumps(
            {'id': time_series_to_get,
             'data':fixed_data[time_series_to_get].tolist()}), 
                                  'utf-8'))
    else:
        handler.send_response(404)
        handler.send_header('Content-type', 'text/html')
        handler.end_headers()
        handler.wfile.write(b"404 not found")
    return


# In[21]:




def do_post(handler):
    #
    # should be /label/42
    if handler.path == '/label':  
        # Construct a server response.
        #print(handler.headers)
        #print(handler.headers['Content-type'])
        
        # TODO: security here -- don't read ourselves out of memory...


        data = handler.rfile.readline()
        #print(data)
        data = json.loads(data.decode('utf-8'))
        #print(data)
        on_new_label(data['id'], data['label'])
        
        handler.send_response(200)
        handler.send_header('Content-type', 'text/html')
        handler.end_headers()
        handler.wfile.write(b"OK")
    else:
        handler.send_response(404)
        handler.send_header('Content-type', 'text/html')
        handler.end_headers()
        handler.wfile.write(b"404 not found")    
        print(f'404 - {handler.path}')
    return


# In[22]:


import http.server
import socketserver
import threading

# try:
#     httpd.shutdown()
#     httpd = None
# except NameError:
#     print('no existing server...')

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        eval('do_get')(self)

    def do_POST(self):
        eval('do_post')(self)

    
def foo():
    global httpd
    port = 8001
    print(f'Server listening on port {port}...')
    httpd = socketserver.TCPServer(('', port), Handler)
    httpd.serve_forever()

t= threading.Thread(target=foo)
t.start()


# In[ ]:





# In[ ]:





# In[ ]:




