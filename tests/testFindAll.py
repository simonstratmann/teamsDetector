import unittest

import findAllNames


class MyTestCase(unittest.TestCase):
    def testFindAll(self):
        findAllNames.doShowImages = False
        self.assertEqual(findAllNames.getSpeakerName("../teamscall sharing speaker no video.png"), "Michael Grotemeyer")
        self.assertEqual(findAllNames.getSpeakerName("../Meeting in jannik speaking.png"), "Janik Schick")
        self.assertEqual(findAllNames.getSpeakerName("../teamscall sharing names.png"), "Hauke Plambeck")


if __name__ == '__main__':
    unittest.main()
