from setuptools import setup, find_packages

setup(
    name='cv_tools',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # 这里添加你的依赖项，例如 'numpy', 'requests' 等
    ],
    entry_points={
        'console_scripts': [
            # 这里添加你的命令行脚本，例如 'your-command=your_module:main'
        ],
    },
)
