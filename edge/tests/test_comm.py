import unittest
from unittest import mock

from src.comm import BandwidthSender


class CommTests(unittest.TestCase):
    def test_worker_disables_env_proxy_by_default(self):
        with mock.patch("src.comm.threading.Thread") as thread_cls:
            sender = BandwidthSender(bandwidth_MBps=1.0)

        thread_cls.return_value.start.assert_called_once()

        sender._q = mock.Mock()
        sender._q.get.side_effect = [None]

        with mock.patch("src.comm.requests.Session") as session_cls:
            session = session_cls.return_value
            sender._worker()

        self.assertFalse(session.trust_env)
        session.close.assert_called_once()

    def test_worker_can_enable_env_proxy_explicitly(self):
        with mock.patch("src.comm.threading.Thread"):
            sender = BandwidthSender(bandwidth_MBps=1.0, use_env_proxy=True)

        sender._q = mock.Mock()
        sender._q.get.side_effect = [None]

        with mock.patch("src.comm.requests.Session") as session_cls:
            session = session_cls.return_value
            sender._worker()

        self.assertTrue(session.trust_env)


if __name__ == "__main__":
    unittest.main()
