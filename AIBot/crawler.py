from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    DefaultMarkdownGenerator,
    PruningContentFilter,
)

# crawl4ai browser configuration
browser_cfg = BrowserConfig(
    browser_type="chromium",
    headless=True,
)

prune_filter = PruningContentFilter(
        # Lower → more content retained, higher → more content pruned
        threshold=0.45,
        # "fixed" or "dynamic"
        threshold_type="dynamic",
        # Ignore nodes with <5 words
        min_word_threshold=5

    )

md_generator = DefaultMarkdownGenerator(content_filter=prune_filter)

run_config = CrawlerRunConfig(
    # Core
    verbose=True,
    cache_mode=CacheMode.BYPASS,
    markdown_generator=md_generator,
    exclude_external_links=True,
    exclude_all_images=True,
    exclude_social_media_links=True,
    excluded_tags=["nav", "footer", "script", "style"],
)

crawler = AsyncWebCrawler(config=browser_cfg, run_config=run_config)