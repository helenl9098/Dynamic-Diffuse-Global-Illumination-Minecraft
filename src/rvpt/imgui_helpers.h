#pragma once

#include <string>

#include <imgui.h>

template <std::size_t count>
inline void dropdown_helper(const char* unique_identifier, int& cur_mode,
                            const char* (&mode_names)[count])
{
    ImGuiStyle& style = ImGui::GetStyle();
    float w = ImGui::CalcItemWidth();
    float spacing = style.ItemInnerSpacing.x;
    float button_sz = ImGui::GetFrameHeight();

    std::string left_arrow_name = std::string("##") + unique_identifier + "_l_arrow";
    if (ImGui::ArrowButton(left_arrow_name.c_str(), ImGuiDir_Left))
    {
        cur_mode = (cur_mode - 1 + count) % count;
    }
    ImGui::SameLine(0, style.ItemInnerSpacing.x);
    std::string right_arrow_name = std::string("##") + unique_identifier + "_r_arrow";
    if (ImGui::ArrowButton(right_arrow_name.c_str(), ImGuiDir_Right))
    {
        cur_mode = (cur_mode + 1 + count) % count;
    }
    ImGui::SameLine(0, style.ItemInnerSpacing.x);
    std::string custom_combo_name = std::string("##") + unique_identifier + "_custom_combo";
    if (ImGui::BeginCombo(custom_combo_name.c_str(), mode_names[cur_mode],
                          ImGuiComboFlags_NoArrowButton))
    {
        for (int n = 0; n < count; n++)
        {
            bool is_selected = (mode_names[cur_mode] == mode_names[n]);
            if (ImGui::Selectable(mode_names[n], is_selected))
            {
                cur_mode = n;
            }
            if (is_selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
}
