import unittest
from unittest import mock

from src.comm import BandwidthSender
from src.software_link import SoftwareLink


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

    def test_software_mode_uploads_before_post_and_downloads_after_response(self):
        events = []
        measurements = []
        link = mock.Mock()

        def transmit(direction, byte_count):
            events.append((direction, byte_count))
            return {"service_seconds": 0.01, "queue_wait_seconds": 0.0}

        link.transmit.side_effect = transmit
        with mock.patch("src.comm.threading.Thread"):
            sender = BandwidthSender(
                bandwidth_MBps=1.0,
                software_bandwidth_emulation=True,
                link=link,
                on_complete=measurements.append,
            )
        future = sender.submit(
            "http://example.test/propose",
            b"payload",
            token_count=1,
            measurement_kind="transport",
        )
        sender._q.put(None)

        response = mock.Mock()
        response.status_code = 200
        response.content = b"reply"
        response.headers = {"Content-Type": "application/json"}
        response.json.return_value = {"ok": True}

        def post(*args, **kwargs):
            self.assertEqual(events, [("uplink", len(b"payload"))])
            events.append(("post", len(kwargs["data"])))
            return response

        with mock.patch("src.comm.requests.Session") as session_cls:
            session_cls.return_value.post.side_effect = post
            sender._worker()

        self.assertEqual(future.result(), {"ok": True})
        self.assertEqual(events, [
            ("uplink", len(b"payload")),
            ("post", len(b"payload")),
            ("downlink", len(b"reply")),
        ])
        self.assertEqual(measurements[0]["upload_seconds"], 0.01)
        self.assertEqual(measurements[0]["download_seconds"], 0.01)
        self.assertEqual(measurements[0]["response_size"], len(b"reply"))
        totals = sender.snapshot()["totals"]
        self.assertEqual(totals["requests"], 1)
        self.assertEqual(totals["successful_requests"], 1)
        self.assertEqual(totals["payload_bytes"], len(b"payload"))
        self.assertEqual(totals["response_bytes"], len(b"reply"))

    def test_json_payload_size_is_serialized_byte_size(self):
        events = []
        link = mock.Mock()
        link.transmit.side_effect = lambda direction, size: (
            events.append((direction, size))
            or {"service_seconds": 0.0, "queue_wait_seconds": 0.0}
        )
        with mock.patch("src.comm.threading.Thread"):
            sender = BandwidthSender(bandwidth_MBps=1.0, link=link)
        future = sender.submit("http://example.test/start", {"task_id": 12})
        sender._q.put(None)
        response = mock.Mock()
        response.status_code = 200
        response.content = b"{}"
        response.headers = {"Content-Type": "application/json"}
        response.json.return_value = {}
        with mock.patch("src.comm.requests.Session") as session_cls:
            session_cls.return_value.post.return_value = response
            sender._worker()
            sent_body = session_cls.return_value.post.call_args.kwargs["data"]
            sent_headers = session_cls.return_value.post.call_args.kwargs["headers"]

        self.assertEqual(future.result(), {})
        self.assertEqual(events[0], ("uplink", len(sent_body)))
        self.assertEqual(sent_headers["Content-Type"], "application/json")

    def test_shared_link_serializes_two_logical_senders(self):
        link = SoftwareLink(
            uplink_MBps=1.0,
            downlink_MBps=10.0,
            uplink_startup_seconds=0.02,
        )
        with mock.patch("src.software_link.time.monotonic", return_value=0.0), mock.patch(
            "src.software_link.time.sleep"
        ):
            first = link.transmit("uplink", 10_000)
            second = link.transmit("uplink", 10_000)

        self.assertAlmostEqual(first["service_seconds"], 0.03)
        self.assertAlmostEqual(second["service_seconds"], 0.03)
        self.assertAlmostEqual(second["queue_wait_seconds"], 0.03)
        snapshot = link.snapshot()
        self.assertEqual(snapshot["totals"]["uplink"]["transfers"], 2)
        self.assertEqual(snapshot["totals"]["uplink"]["bytes"], 20_000)

    def test_bandwidth_profile_changes_at_configured_interval(self):
        with mock.patch("src.software_link.time.monotonic", return_value=0.0):
            link = SoftwareLink(
                uplink_MBps=2.5,
                downlink_MBps=25.0,
                bandwidth_profile=[(1.25, 18.75), (10.0, 35.0)],
                profile_interval_seconds=20.0,
                profile_started_at=0.0,
            )
        with mock.patch("src.software_link.time.monotonic", return_value=21.0), mock.patch(
            "src.software_link.time.sleep"
        ):
            transfer = link.transmit("uplink", 10_000)
        self.assertEqual(transfer["profile_index"], 1)
        self.assertEqual(transfer["bandwidth_MBps"], 10.0)


if __name__ == "__main__":
    unittest.main()
