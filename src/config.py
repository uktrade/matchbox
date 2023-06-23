import splink.duckdb.comparison_library as cl
import splink.duckdb.comparison_template_library as ctl

settings = {
    "link_type": "link_and_dedupe",
    "retain_matching_columns": True,
    "retain_intermediate_calculation_columns": True,
    "blocking_rules_to_generate_predictions": [
        """
            ((l.comp_num_clean = r.comp_num_clean))
            and (
                l.comp_num_clean <> ''
                and r.comp_num_clean <> ''
            )
        """,
        """
            (l.name_unusual_tokens = r.name_unusual_tokens)
            and (
                l.name_unusual_tokens <> ''
                and r.name_unusual_tokens <> ''
            )
        """,
        # """
        #     (l.name_unusual_tokens_first5 = r.name_unusual_tokens_first5)
        #     and (
        #         length(l.name_unusual_tokens_first5) = 5
        #         and length(r.name_unusual_tokens_first5) = 5
        #     )
        # """,
        # """
        #     (l.name_unusual_tokens_last5 = r.name_unusual_tokens_last5)
        #     and (
        #         length(l.name_unusual_tokens_last5) = 5
        #         and length(r.name_unusual_tokens_last5) = 5
        #     )
        # """,
        """
            (l.secondary_name_unusual_tokens = r.secondary_name_unusual_tokens)
            and (
                l.secondary_name_unusual_tokens <> ''
                and r.secondary_name_unusual_tokens <> ''
            )
        """,
        """
            (l.secondary_name_unusual_tokens = r.name_unusual_tokens)
            and (
                l.secondary_name_unusual_tokens <> ''
                and r.name_unusual_tokens <> ''
            )
        """,
        """
            (r.secondary_name_unusual_tokens = l.name_unusual_tokens)
            and (
                r.secondary_name_unusual_tokens <> ''
                and l.name_unusual_tokens <> ''
            )
        """,
        # My attempt to reduce computation on first/last 5 while retaining info
        # """
        #     (l.name_sig_first5 = r.name_sig_first5)
        #     and (
        #         length(l.name_sig_first5) = 5
        #         and length(r.name_sig_first5) = 5
        #     )
        # """,
        # """
        #     (l.name_sig_last5 = r.name_sig_last5)
        #     and (
        #         length(l.name_sig_last5) = 5
        #         and length(r.name_sig_last5) = 5
        #     )
        # """,
        # TODO: blocking rule on first token name_unusual_tokens?
    ],
    # for comp_num_clean: there may be some typos
    # for name_unusual_tokens: may be some typos but also a lot of 'duplicate ...' gumph
    #   hence two different similarity levels
    "comparisons": [
        cl.jaro_winkler_at_thresholds(
            "comp_num_clean", [0.75], term_frequency_adjustments=True
        ),
        cl.jaro_winkler_at_thresholds(
            "name_unusual_tokens", [0.9, 0.6], term_frequency_adjustments=True
        ),
        # match on the first alphabetic chars in the postcode
        # TODO: change to geographic similarity measure?
        # cl.exact_match("postcode_area", 2),
        ctl.postcode_comparison("postcode")
        # TODO: try a comparison on main name and secondary name - how to do?
        # Add first name to secondary name array
        # Use ct.array_intersect_at_sizes?
        # TODO: secondary_name_unusual_tokens comparison
        # cl.array_intersect_at_sizes("alternative_company_names", [1])
    ],
}

stopwords = [
    "limited",
    "uk",
    "company",
    "international",
    "group",
    "of",
    "the",
    "inc",
    "and",
    "plc",
    "corporation",
    "llp",
    "pvt",
    "gmbh",
    "u k",
    "pte",
    "usa",
    "bank",
    "b v",
    "bv",
]
