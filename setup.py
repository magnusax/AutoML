try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
  name = 'gazer',
  packages = ['gazer'], # this must be the same as the name above
  version = '0.1',
  description = 'Machine learning library built on top of several popular projects, e.g., scikit-learn and scikit-optimize.',
  author = 'Magnus Axelsson',
  author_email = 'johanmagnusaxelsson@gmail.com',
  url = 'https://github.com/magnusax/ml-meta-wrapper', # use the URL to the github repo
  keywords = ['machine learning', 'software'], # arbitrary keywords
  install_requires=["numpy", "scipy", "scikit-learn>=0.17", "seaborn", "skopt>=0.3"]
  classifiers = [],
)