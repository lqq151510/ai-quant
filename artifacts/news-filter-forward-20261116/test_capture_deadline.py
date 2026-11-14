"""Exercise provider failures and fallback through a loopback HTTP connection."""

import hashlib
import json
import signal
import sys
import unittest
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from unittest.mock import patch
from urllib.request import ProxyHandler, build_opener

import collect_books
from test_book_evidence import DAY, STOCK, tencent_row


class CaptureDeadlineTests(unittest.TestCase):
    def test_failed_responses_have_no_receipt_and_fall_back_to_healthy_provider(self):
        healthy = tencent_row(STOCK, 9).encode('gb18030')
        completed_slow_body = Event()
        stop_slow_body = Event()

        class Provider(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def do_GET(self):
                self.send_response(200)
                self.send_header('Connection', 'close')
                if self.path.startswith('/slow/'):
                    self.send_header('Content-Length', '24')
                    self.end_headers()
                    try:
                        for _ in range(24):
                            self.wfile.write(b'x')
                            if stop_slow_body.wait(.04):
                                return
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    completed_slow_body.set()
                elif self.path.startswith('/short/'):
                    self.send_header('Content-Length', '10')
                    self.end_headers()
                    self.wfile.write(b'short')
                elif self.path.startswith('/chunked/'):
                    self.send_header('Transfer-Encoding', 'chunked')
                    self.end_headers()
                    self.wfile.write(b'10\r\nx')
                else:
                    self.send_header('Content-Length', str(len(healthy)))
                    self.end_headers()
                    self.wfile.write(healthy)

            def log_message(self, _format, *_args):
                pass

        class Clock(datetime):
            @classmethod
            def now(cls, tz):
                return cls.fromisoformat(f'{DAY}T09:45:10+08:00').astimezone(tz)

        capture = collect_books.capture
        opener = build_opener(ProxyHandler({}))
        previous_handler = signal.getsignal(signal.SIGALRM)
        with ThreadingHTTPServer(('127.0.0.1', 0), Provider) as server:
            serving = Thread(target=server.serve_forever, kwargs={'poll_interval': .02}, daemon=True)
            serving.start()
            base_url = f'http://127.0.0.1:{server.server_port}/'

            def capture_with_short_test_deadline(url, _timeout):
                return capture(url, timeout=2 if url.startswith(base_url + 'healthy/') else .3)

            try:
                for failure, reason in (('slow', 'deadline'), ('short', 'IncompleteRead'),
                                        ('chunked', 'IncompleteRead')):
                    with self.subTest(failure=failure), TemporaryDirectory() as directory:
                        output = Path(directory)
                        sources = {'sina': base_url + failure + '/', 'tencent': base_url + 'healthy/'}
                        args = ['collect_books.py', '--date', str(DAY), '--out', directory, STOCK]
                        with (patch.object(collect_books, 'SOURCES', sources),
                              patch.object(collect_books, 'datetime', Clock),
                              patch.object(collect_books, 'urlopen', opener.open),
                              patch.object(collect_books, 'capture', capture_with_short_test_deadline),
                              patch.object(sys, 'argv', args)):
                            collect_books.main()

                        result = json.loads((output / 'books.json').read_text())
                        first, fallback = result['attempts']
                        self.assertEqual(set(first), {'source', 'url', 'error'})
                        self.assertEqual(first['source'], 'sina')
                        self.assertIn(reason, first['error'])
                        self.assertEqual(fallback['source'], 'tencent')
                        self.assertEqual(fallback['retrieved_at'], f'{DAY}T09:45:10+08:00')
                        self.assertEqual(fallback['raw_sha256'], hashlib.sha256(healthy).hexdigest())
                        self.assertEqual(result['selected'][STOCK]['raw_file'], '002-tencent.txt')
                        self.assertEqual(list(output.glob('*.txt')), [output / '002-tencent.txt'])
                        self.assertEqual((output / '002-tencent.txt').read_bytes(), healthy)
                        self.assertNotIn('pending_response', json.loads((output / 'capture-progress.json').read_text()))
                        self.assertIs(signal.getsignal(signal.SIGALRM), previous_handler)
                        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))
                        if failure == 'slow':
                            self.assertFalse(completed_slow_body.is_set())
            finally:
                stop_slow_body.set()
                server.shutdown()
                serving.join()
