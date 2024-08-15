import unittest

import boxhead as BH_py

class TestBoxHead(unittest.TestCase):
    def testCharActionExists(self):
        self.assertIsNotNone(BH_py.CharAction)

    def testBoxHeadClassExists(self):
        self.assertIsNotNone(BH_py.BoxHead)

    def testPygameInitExists(self):
        self.assertIsNotNone(BH_py.BoxHead._pygame_initialise)

    def testResetExists(self):
        self.assertIsNotNone(BH_py.BoxHead.reset)


unittest.main()