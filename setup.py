from setuptools import setup, find_packages

setup(
  name = 'local-attention',
  packages = find_packages(),
  version = '1.7.1',
  license='MIT',
  description = 'Local attention, window with lookback, for language modeling',
  long_description_content_type = 'text/markdown',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/local-attention',
  keywords = [
    'transformers',
    'attention',
    'artificial intelligence'
  ],
  install_requires=[
    'einops>=0.6.0',
    'torch'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
