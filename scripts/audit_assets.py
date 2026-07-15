#!/usr/bin/env python
from kinasephos.cli import main

if __name__ == "__main__":
    main(["audit-assets", *__import__("sys").argv[1:]])
