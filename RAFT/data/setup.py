from setuptools import setup, find_packages

setup(
    name='hpcbot',  
    version='0.1.0',  
    description='Tool to generate artificial QA',  
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/btrungvo/hpcbot',
    author='Trung Vo',
    author_email='trungvo.usth@gmail.com',
    license='MIT',  
    packages=find_packages(),
    install_requires=[
        'langchain_community',
        "langchain-text-splitters",
        'openai'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)