from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='NiChartHarmonize',
        version='2.2.0',
        description='Harmonization tools for multi-center neuroimaging studies.',
        long_description=readme(),
        url='https://github.com/rpomponiohttps://github.com/gurayerus/NiChartHarmonize',
        author='Guray Erus',
        author_email='guray.erus@pennmedicine.upenn.edu',
        license='MIT',
        packages=['NiChartHarmonize'],
        install_requires=['numpy', 'pandas', 'nibabel', 'statsmodels>=0.12.0'],
        entry_points={
            'console_scripts': ['neuroharm=NiChartHarmonize.nh_cli:main']
            },
        zip_safe=False)
