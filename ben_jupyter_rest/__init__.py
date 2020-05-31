def _jupyter_server_extension_paths():
    return [{
        "module": "ben_jupyter_rest"
    }]

from notebook.utils import url_path_join
from notebook.base.handlers import IPythonHandler

class HelloWorldHandler(IPythonHandler):
    def get(self):
        self.finish('Hello, world!')


def load_jupyter_server_extension(nbapp):
    nbapp.log.info("ben's module enabled!")

    web_app = nbapp.web_app
    host_pattern = '.*$'
    route_pattern = url_path_join(web_app.settings['base_url'], '/hello')
    web_app.add_handlers(host_pattern, [(route_pattern, HelloWorldHandler)])



