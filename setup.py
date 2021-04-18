from setuptools import setup, find_packages

setup(
  name = 'local-attention',
  packages = find_packages(),
  version = '1.4.1',
  license='MIT',
  description = 'Local windowed attention, for language modeling',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/local-attention',
  keywords = ['transformers', 'attention', 'artificial intelligence'],
  install_requires=[
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