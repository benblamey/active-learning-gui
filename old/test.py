#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import math


# In[ ]:


# GET /convert
req = json.loads(REQUEST)
args = req['args']

if 'angle' not in args:
  print(json.dumps({'convertedAngle': None}))
else:
  # Note the [0] when retrieving the argument.
  # This is because you could potentially pass multiple angles.
  angle = int(args['angle'][0])
  converted = math.radians(angle)
  print(json.dumps({'convertedAngle': converted}))


# In[1]:



import http.server
import socketserver



import threading

class Handler(http.server.SimpleHTTPRequestHandler):

    def do_GET(self):

        # Construct a server response.
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"hej")
        
        return


def foo():
    print('Server listening on port 8000...')
    httpd = socketserver.TCPServer(('', 8000), Handler)
    httpd.serve_forever()

t= threading.Thread(target=foo)
t.start()


# In[2]:


print('hej')


# In[2]:


1


# In[3]:



import threading
from IPython.display import display
import ipywidgets as widgets
import time

def get_ioloop():
    import IPython, zmq
    ipython = IPython.get_ipython()
    if ipython and hasattr(ipython, 'kernel'):
        return zmq.eventloop.ioloop.IOLoop.instance()


#The IOloop is shared
ioloop = get_ioloop()

print(ioloop)


# In[ ]:




