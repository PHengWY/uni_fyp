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

    def testRenderExists(self):
        self.assertIsNotNone(BH_py.BoxHead.render)

    def testEventExists(self):
        self.assertIsNotNone(BH_py.BoxHead._emergency_event)

    def testBackgroundExists(self):
        self.assertIsNotNone(BH_py.BoxHead.get_background)

    def testBackgroundDrawn(self):
        self.assertIsNotNone(BH_py.BoxHead._draw_background)

    # def testWallDimensions(self):
    #     game = BH_py.BoxHead() 

    #     first_wall = game._draw_walls[0]
    #     self.assertGreaterEqual(first_wall.width, 10)
    #     self.assertLessEqual(first_wall.width, 50)
    #     self.assertGreaterEqual(first_wall.height, 20) 
    #     self.assertLessEqual(first_wall.height, 80)

    


unittest.main()