from setuptools import setup

setup(name='toolboxds',
      version='0.1',
      description='Tool box para cientista de dados',
      url='https://github.com/lourencoerick/toolboxds',
      author='Erick Lourenco',
      author_email='lourenco.erick@gmail.com',
      license='MIT',
      packages=['toolboxds'],
      install_requires=['markdown', 'plotly', 'numpy', 'pandas', 'scipy', 'sklearn'],
      zip_safe=False)
