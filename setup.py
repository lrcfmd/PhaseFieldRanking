from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='ranking_phase_fields',
      version='0.1',
      description='Ranking phase fields with likelihood of finding a stable composition',
      url='http://github.com/DrewNow/RankingPhaseFields',
      author='Andrij Vasylenko',
      author_email='and.vasylenko@gmail.com',
      license='MIT',
      packages=['ranking_phase_fields'],
      install_requires=['pyod'],
      include_package_data=True,
      entry_points={"console_scripts": ["ranking_phase_fields=ranking_phase_fields.__main__:main"]},  
      zip_safe=False)
