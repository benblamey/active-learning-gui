#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import http.server
# import socketserver

# PORT = 8002

# Handler = http.server.SimpleHTTPRequestHandler

# class S(http.server.BaseHTTPRequestHandler):
#     def do_GET(self):
#         logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
#         print('foo')
#         self._set_response()
#         self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))
    

# with socketserver.TCPServer(("", PORT), S) as httpd:
#     print("serving at port", PORT)
#     httpd.serve_forever()


# In[2]:


import http.server
import socketserver

PORT = 8001

Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()


# In[ ]:


print(f;;)

