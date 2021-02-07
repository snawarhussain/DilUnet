try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
config = {

    'description': 'Project',
    'author': 'Snawar',
    'url': 'no',
    'download_url': 'no',
    'author_email': 'sanawar.hussain18@gmail.com',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': ['ex47'],
    'scripts': [],
    'name': 'automated_testing'


}

setup(**config)