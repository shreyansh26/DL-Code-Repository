  
from setuptools import setup

setup(name='tfb',
      version='0.1',
      description="Basic Transformer Implementation based on Peter Bloem's blog",
      url='http://www.peterbloem.nl/blog/transformers',
      author='Shreyansh Singh',
      author_email='shreyansh.pettswood@gmail.com',
      license='MIT',
      packages=['tfb'],
      install_requires=[
            'torch',
            'tqdm',
            'numpy',
            'torchtext'
      ],
      zip_safe=False)