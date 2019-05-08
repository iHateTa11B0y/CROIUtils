from setuptools import setup

setup(
        name='maskrcnn_utils',
        version='0.1.0',
        description='this is a tool package for maskrcnn',
        author='Kuuuurt',
        author_email='findme@somewhere.com',
        url='haishentec.com',
        packages=['maskrcnn_utils'],
        package_data={'maskrcnn_utils': ['_C.so'],},
)


