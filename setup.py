from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='rankfields',
      version='0.1.2.4',
      description='Ranking phase fields with likelihood of finding a stable composition',
      url='http://github.com/lrcfmd/RankingPhaseFields',
      author='Andrij Vasylenko',
      author_email='and.vasylenko@gmail.com',
      license='MIT',
      packages=['ranking_phase_fields'],
      scripts=['rankfields'],
      install_requires=['pyod','numpy','tensorflow'],
#      python_requires='>=3.7, <3.9',
      include_package_data=True,
      package_data={'': ['ranking_phase_fields/rpp.input']},
      entry_points={"console_scripts": ["ranking_phase_fields=ranking_phase_fields.__main__:main"]},  
      zip_safe=False)
