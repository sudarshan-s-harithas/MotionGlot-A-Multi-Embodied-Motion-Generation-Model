"""Exercise all three human tasks in Chrome, plus playback and mobile layout.

pip install playwright; run with --url http://127.0.0.1:7862 and
--model-base http://127.0.0.1:7862/models/ for the local export server.
"""
import argparse
import asyncio
import json
from pathlib import Path
from playwright.async_api import async_playwright


async def check(args):
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=args.chrome, headless=True,
            args=['--no-sandbox', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        page = await browser.new_page(viewport={'width': 1440, 'height': 1100})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        if args.model_base:
            await page.route('**/settings.json', lambda route: route.fulfill(json={'modelBase': args.model_base}))
        await page.goto(args.url, wait_until='networkidle', timeout=60000)
        await page.wait_for_function("document.querySelector('#download').disabled === false", timeout=60000)
        assert await page.locator('#task option').count() == 3
        assert 'robot' not in (await page.locator('body').inner_text()).lower()
        results = []
        # A current motion stays selected when switching to its captioning task.
        for task in ['text-to-human', 'question-to-human', 'human-to-text']:
            await page.select_option('#task', task)
            await page.click('#generate')
            await page.wait_for_function("document.querySelector('#generate').disabled === false", timeout=180000)
            status = await page.locator('#status').inner_text()
            assert not await page.locator('#status').evaluate("e => e.classList.contains('error')"), (task, status)
            assert 'End marker reached' in status, (task, status)
            assert await page.locator('#origin').inner_text() == 'GENERATED IN YOUR BROWSER'
            if task.endswith('-to-text'):
                assert (await page.locator('#caption').inner_text()).strip()
            results.append({'task': task, 'status': status, 'caption': await page.locator('#caption').inner_text()})
            print(json.dumps(results[-1]), flush=True)
        # Read the pinned adapter's actual MuJoCo state on local development hosts.
        if '127.0.0.1' in args.url or 'localhost' in args.url:
            state = await page.evaluate('''(()=>{const v=motionglotDebug.viewer;return {
                time:v.runtime.mjData.time,running:v.runtime.running,
                renderedBodies:v.runtime.lastSimState.bodies.size,
                mocapBodies:v.runtime.mjModel.nmocap};})()''')
            assert state['time'] == 0 and state['running'] is False, state
            assert state['renderedBodies'] == state['mocapBodies'], state
            results.append({'physics': state})
        await page.locator('.example-card').first.click()
        await page.wait_for_timeout(300)
        await page.screenshot(path=str(args.output / 'desktop.png'), full_page=True)
        await page.set_viewport_size({'width': 390, 'height': 844})
        await page.wait_for_timeout(400)
        assert await page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        await page.screenshot(path=str(args.output / 'mobile.png'), full_page=True)
        assert not errors, errors
        (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        await browser.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:7862/')
    parser.add_argument('--model-base')
    parser.add_argument('--chrome', default='/usr/bin/google-chrome')
    parser.add_argument('--output', type=Path, default=Path('local/browser-qa'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    asyncio.run(check(args))
