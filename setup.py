from setuptools import setup, find_packages

setup(
  name='gazer',
  version='0.1.dev1',
  description='Machine learning library built on top of several popular projects, e.g., scikit-learn and scikit-optimize.',
  author='Magnus Axelsson',
  author_email='johanmagnusaxelsson@gmail.com',
  url='https://github.com/magnusax/AutoML', # use the URL to the github repo
  keywords=['machine learning', 'software'], # arbitrary keywords
  install_requires=["numpy", "scipy", "scikit-learn>=0.17", 
                    "scikit-optimize>=0.3", "joblib", "tqdm"],
  license='MIT License',
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development',
    'Topic :: Scientific/Engineering',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6'],
    
  packages=find_packages(),
  python_requires='>=3.4',
)
