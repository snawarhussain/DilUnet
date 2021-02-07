import unittest



from ex47.game import Room
class TestSum(unittest.TestCase):
    def test_room(self):
        gold = Room('GoldRoom', """This room has gold in it , you can grab. There's a
        door to the north.""")
        self.assertEqual(gold.name, 'GoldRoom')
        assert gold.paths == {}

    def test_room_path(self):
        center = Room('Center', "test room in the center")
        north = Room('North', 'test room in the north')
        south = Room('South', 'test room in the south')
        center.add_path({'north': north, 'south': south})
        self.assertEqual(center.go('north'), north)
        self.assertEqual(center.go('south'), south)
    def test_basic(self):
        print("I RAN!")

if __name__ == '__main__':
    unittest.main()