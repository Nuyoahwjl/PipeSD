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

    def test_os_shaping_mode_does_not_add_post_response_sleep(self):
        measurements = []
        with mock.patch("src.comm.threading.Thread"):
            sender = BandwidthSender(
                bandwidth_MBps=0.000001,
                base_latency=10.0,
                software_bandwidth_emulation=False,
                on_complete=measurements.append,
            )
        future = sender.submit(
            "http://example.test/propose",
            b"payload",
            token_count=2,
            measurement_kind="transport",
        )
        sender._q.put(None)

        response = mock.Mock()
        response.status_code = 200
        response.headers = {"Content-Type": "application/json"}
        response.json.return_value = {"ok": True}
        with mock.patch("src.comm.requests.Session") as session_cls, mock.patch(
            "src.comm.time.sleep"
        ) as sleep:
            session_cls.return_value.post.return_value = response
            sender._worker()

        self.assertEqual(future.result(), {"ok": True})
        sleep.assert_not_called()
        self.assertEqual(measurements[0]["measurement_kind"], "transport")


if __name__ == "__main__":
    unittest.main()
