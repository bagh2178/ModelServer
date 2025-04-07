from setuptools import setup, find_packages

setup(
    name='ModelServer',
    version='0.2.0',
    description='A framework for reading and controling robots remotely.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Hang Yin',
    author_email='yinh23@mails.tsinghua.edu.cn',
    url='https://github.com/bagh2178/ModelServer',
    license='MIT',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[],
)