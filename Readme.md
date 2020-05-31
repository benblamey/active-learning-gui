
Setup: 

```
≈
jupyter kernelgateway --generate-config
jupyter kernelgateway --api=kernel_gateway.notebook_http --seed_uri=$(pwd)/Vilar_Oscillator/Vilar_OscillatorSciopeMI.ipynb kernelgateway --api=kernel_gateway.notebook_http

jupyter kernelgateway --api='kernel_gateway.notebook_http' --seed_uri='./Vilar_Oscillator/test.ipynb'
jupyter kernelgateway --api=kernel_gateway.notebook_http --seed_uri=$(pwd)/Vilar_Oscillator/test.ipynb --KernelGatewayApp.force_kernel_name=python3

jupyter kernelgateway --KernelGatewayApp.api='kernel_gateway.notebook_http' --KernelGatewayApp.seed_uri='/home/username/Notebook.ipynb'
```



# install from pypi
pip install jupyter_kernel_gateway

# show all config options
jupyter kernelgateway --help-all

# run it with default options
jupyter kernelgateway


Notes to self:

https://github.com/jupyter/kernel_gateway
https://ndres.me/post/jupyter-notebook-rest-api/
https://jupyter-notebook.readthedocs.io/en/stable/extending/handlers.html
https://github.com/suvarchal/nbapp
https://gist.github.com/the-moog/94b09b49232731bd2a3cedd24501e23b

... we get the API instead of the web GUI, not as well.
... this is intended for a develop then deploy use case
... this is not exactly what we're trying to do

select parameter points which:
- exploit area of known 'good' class
	- sub-sample of known 'good' class
- uncertainty of the classifier
- unexplored points in the parameter space

TODO:
- optimize distribution of tables
- down stream analysis... spark cluster?


-----


START OVER, JUPYTER EXTENSION...

jupyter serverextension enable --py jupyter_rest


...hmmm... with this, I'm in the Tordado web app -- not the Kernel...


-----

START OVER, RUN IT INSIDE THE CELL...

def run(server_class=HTTPServer, handler_class=BaseHTTPRequestHandler):
    server_address = ('', 8000)
    httpd = server_class(server_address, handler_class)
    httpd.serve_forever()
    
    
    
    
https://www.tornadoweb.org/en/stable/asyncio.html