from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='ranking_phase_fields',
      version='0.1',
      description='Ranking wrt likelihood of finding a stable composition',
      url='http://github.com/DrewNow/RankingPhaseFields',
      author='Andrij Vasylenko',
      author_email='and.vasylenko@gmail.com',
      license='MIT',
      packages=['ranking_phase_fields'],
      install_requires=[
          'pyod',
      ],
      zip_safe=False)
