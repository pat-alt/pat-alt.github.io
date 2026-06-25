function Meta(meta)
	if meta.propositions then
		local propositions_file = pandoc.utils.stringify(meta.propositions)

		-- Read the propositions file
		local file = io.open(propositions_file, "r")
		if file then
			local content = file:read("*all")
			file:close()

			-- Process the content through Pandoc to convert Markdown to LaTeX
			local doc = pandoc.read(content, "markdown")

			-- Remove any headers from the content and log if found
			local filtered_blocks = {}
			local found_headers = false
			for _, block in ipairs(doc.blocks) do
				if block.t == "Header" then
					found_headers = true
					local header_text = pandoc.utils.stringify(block.content)
					io.stderr:write(
						"INFO: Ignoring header '"
							.. header_text
							.. "' in "
							.. propositions_file
							.. " (using default 'Propositions' title)\n"
					)
				else
					table.insert(filtered_blocks, block)
				end
			end

			-- Create new document with filtered blocks
			local filtered_doc = pandoc.Pandoc(filtered_blocks)
			local latex_content = pandoc.write(filtered_doc, "latex")

			-- Set the processed content as a template variable
			meta.propositions = pandoc.RawBlock("latex", latex_content)
		else
			-- Handle file not found
			meta.propositions = pandoc.RawBlock("latex", "% Propositions file not found: " .. propositions_file)
		end
	end

	return meta
end
