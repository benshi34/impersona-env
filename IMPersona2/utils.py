"""
Impersona-Env — Shared Utilities
================================

Utility functions used across the Impersona-Env pipeline.
"""

import re


def normalize_phone(phone):
    """Normalize phone numbers to a standard format by removing non-digits."""
    return ''.join(c for c in phone if c.isdigit())


def parse_vcard_file(file_path):
    """Parse vCard file and return a dictionary mapping phone numbers to names."""
    phone_to_name = {}
    current_name = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Get the display name
            if line.startswith('FN:'):
                current_name = line[3:].strip()

            # Get phone numbers
            elif line.startswith('TEL') or line.startswith('item1.TEL'):
                # Extract phone number from the line
                phone = line.split(':')[-1].strip()

                # Normalize the phone number
                normalized_phone = normalize_phone(phone)

                # Only add if we have both a name and a valid phone number
                if current_name and normalized_phone:
                    # Remove any "mobile" label or other metadata
                    if len(normalized_phone) >= 10:  # Only store valid phone numbers
                        phone_to_name[normalized_phone] = current_name

            # Reset current name at the end of a vCard entry
            elif line == 'END:VCARD':
                current_name = None

    return phone_to_name
